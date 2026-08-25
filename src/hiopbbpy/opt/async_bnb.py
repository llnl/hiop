"""Master-side state for certified asynchronous spatial branch-and-bound.

This module deliberately contains no GP or relaxation code.  Workers compute two
child bounds; the master owns the partition, incumbent, node states, pruning,
and the global lower-bound certificate.
"""

from __future__ import annotations

import copy
import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class LeafState(str, Enum):
  READY = "ready"
  INFLIGHT = "inflight"
  CLOSED = "closed"


class CloseReason(str, Enum):
  PRUNED = "pruned"
  LOCAL_GAP = "local_gap"


@dataclass
class BnBNode:
  """A single leaf record.

  The first four arguments preserve the constructor used by the existing
  ``bnbalgorithm.py`` implementation.
  """

  l: np.ndarray
  u: np.ndarray
  aq_L: float
  aq_U: float
  aq_U_x: Optional[np.ndarray] = None
  node_id: Optional[int] = None
  parent_id: Optional[int] = None
  depth: int = 0
  state: LeafState = LeafState.READY
  close_reason: Optional[str] = None
  generation: int = 0
  metadata: Dict[str, Any] = field(default_factory=dict)

  # Retained for compatibility with current diagnostics.
  parent_aq_U: Optional[float] = None
  parent_aq_L: Optional[float] = None
  parent_aq_U_x: Optional[np.ndarray] = None

  def __post_init__(self) -> None:
    self.l = np.asarray(self.l, dtype=float).copy()
    self.u = np.asarray(self.u, dtype=float).copy()
    self.aq_L = float(np.asarray(self.aq_L).reshape(-1)[0])
    self.aq_U = float(np.asarray(self.aq_U).reshape(-1)[0])
    if self.aq_U_x is not None:
      self.aq_U_x = np.asarray(self.aq_U_x, dtype=float).reshape(-1).copy()
    if self.parent_aq_U_x is not None:
      self.parent_aq_U_x = np.asarray(self.parent_aq_U_x, dtype=float).reshape(-1).copy()
    if isinstance(self.state, str):
      self.state = LeafState(self.state)
    if self.l.ndim != 1 or self.u.ndim != 1 or self.l.shape != self.u.shape:
      raise ValueError("BnB node corners must be one-dimensional arrays of equal shape")
    if np.any(self.l > self.u):
      raise ValueError("BnB node has a lower corner above its upper corner")
    if math.isnan(self.aq_L) or math.isnan(self.aq_U):
      raise ValueError("BnB bounds must not be NaN")
    if self.aq_U_x is not None and self.aq_U_x.shape != self.l.shape:
      raise ValueError("aq_U_x has the wrong dimension")

  @property
  def diam(self) -> float:
    return float(np.max(self.u - self.l))

  @property
  def midpoint(self) -> np.ndarray:
    return 0.5 * (self.l + self.u)

  @property
  def volume(self) -> float:
    return float(np.prod(self.u - self.l))

  def clone(self) -> "BnBNode":
    return copy.deepcopy(self)

  def __lt__(self, other: "BnBNode") -> bool:
    # Heap ordering is explicit elsewhere.  Keep a deterministic fallback.
    lhs = (self.aq_L, -self.depth, -1 if self.node_id is None else self.node_id)
    rhs = (other.aq_L, -other.depth, -1 if other.node_id is None else other.node_id)
    return lhs < rhs


@dataclass
class BranchResult:
  """Result returned by one worker task for one parent leaf."""

  parent_id: int
  generation: int
  children: Tuple[BnBNode, ...] = field(default_factory=tuple)
  error: Optional[str] = None
  worker_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InFlightRecord:
  """Master-side record for a submitted parent task.

  ``task_id`` is the parent ID because the current MPIEvaluator does not expose
  its Future objects.  The worker echoes the parent ID and generation.
  """

  task_id: int
  parent_id: int
  generation: int
  submitted_at: float
  incumbent_at_submit: float
  attempt: int
  metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncLeafPartition:
  """Authoritative partition and search-state index owned by the master."""

  def __init__(
      self,
      epsilon_prune: float,
      epsilon_node: float,
      bound_consistency_tol: float = 1.0e-4,
      debug_checks: bool = False,
  ) -> None:
    if epsilon_prune < 0.0 or epsilon_node < 0.0:
      raise ValueError("BnB tolerances must be nonnegative")
    self.epsilon_prune = float(epsilon_prune)
    self.epsilon_node = float(epsilon_node)
    self.bound_consistency_tol = float(bound_consistency_tol)
    self.debug_checks = bool(debug_checks)

    self.leaves: Dict[int, BnBNode] = {}
    self.ready: List[Tuple[float, int, int]] = []
    self.inflight: Dict[int, InFlightRecord] = {}

    # An ID-only lazy heap over every leaf gives the global lower bound while
    # retaining the Section-2 rule that each leaf record is stored only once.
    self._all_leaf_lower_bounds: List[Tuple[float, int]] = []
    self._next_node_id = 0
    self.generation = 0
    
    # Least upper bound is the incumbent
    self.incumbent_value = math.inf
    self.incumbent_x: Optional[np.ndarray] = None
    self.incumbent_leaf_id: Optional[int] = None

    self.accepted_parent_tasks = 0
    self.stale_results = 0

  # ------------------------------------------------------------------------
  # Construction and indexing
  # ------------------------------------------------------------------------
  def _allocate_id(self) -> int:
    node_id = self._next_node_id
    self._next_node_id += 1
    return node_id

  def _prepare_id(self, leaf: BnBNode) -> None:
    if leaf.node_id is None:
      leaf.node_id = self._allocate_id()
    else:
      leaf.node_id = int(leaf.node_id)
      self._next_node_id = max(self._next_node_id, leaf.node_id + 1)

  def _push_ready(self, leaf: BnBNode) -> None:
    if leaf.node_id is None:
      raise ValueError("Cannot index a leaf without an ID")
    heapq.heappush(self.ready, (leaf.aq_L, -leaf.depth, leaf.node_id))

  def _push_all_leaf_lb(self, leaf: BnBNode) -> None:
    if leaf.node_id is None:
      raise ValueError("Cannot index a leaf without an ID")
    heapq.heappush(self._all_leaf_lower_bounds, (leaf.aq_L, leaf.node_id))

  def _insert_leaf(self, leaf: BnBNode) -> None:
    if leaf.node_id is None:
      raise ValueError("Cannot insert a leaf without an ID")
    if leaf.node_id in self.leaves:
      raise ValueError("Duplicate BnB node ID: %s" % leaf.node_id)
    self.leaves[leaf.node_id] = leaf
    self._push_all_leaf_lb(leaf)
    if leaf.state == LeafState.READY:
      self._push_ready(leaf)

  def initialize_root(self, root: BnBNode) -> None:
    self.leaves.clear()
    self.ready.clear()
    self.inflight.clear()
    self._all_leaf_lower_bounds.clear()
    self._next_node_id = 0
    self.accepted_parent_tasks = 0
    self.stale_results = 0

    root = root.clone()
    root.parent_id = None
    root.depth = 0
    root.generation = self.generation
    root.close_reason = None
    self._prepare_id(root)

    self.incumbent_value = root.aq_U
    self.incumbent_x = None if root.aq_U_x is None else root.aq_U_x.copy()
    self.incumbent_leaf_id = root.node_id if self.incumbent_x is not None else None
    self._classify(root)
    self._insert_leaf(root)
    if self.debug_checks:
      self.assert_invariants()

  def _classify(self, leaf: BnBNode) -> LeafState:
    """
    Classify a leaf node as READY or CLOSED when new nodes are added, on BnB restart,
    or after the incumbent decreases. It does not deal with/classify as INFLIGHT.
    """
    leaf.close_reason = None
    if leaf.aq_L >= self.incumbent_value - self.epsilon_prune:
      leaf.state = LeafState.CLOSED
      leaf.close_reason = CloseReason.PRUNED.value
    elif leaf.aq_U - leaf.aq_L <= self.epsilon_node:
      leaf.state = LeafState.CLOSED
      leaf.close_reason = CloseReason.LOCAL_GAP.value
    else:
      leaf.state = LeafState.READY
    return leaf.state

  def _reclassify_ready_after_incumbent_update(self) -> int:
    """
    Prune/close READY leaves after the incumbent decreases.

    In-flight parents remain in-flight until their returned children atomically
    replace them.  Those children are classified using the newest incumbent.
    """
    newly_closed = 0
    for leaf in self.leaves.values():
      if leaf.state != LeafState.READY:
        continue
      old_state = leaf.state
      self._classify(leaf)
      if old_state == LeafState.READY and leaf.state == LeafState.CLOSED:
        newly_closed += 1
    return newly_closed

  def global_lower_bound(self) -> float:
    """
    Returns the global lower bound from the heap of leaf lower bounds
    """
    while self._all_leaf_lower_bounds:
      lower_bound, node_id = self._all_leaf_lower_bounds[0]
      leaf = self.leaves.get(node_id)

      # the _all_leaf_lower_bounds heap is lazy (not always synced with self.leaves) and
      # it may be that
      #  i. node was removed from self.leaves, in which case leaf is None or
      # ii. same node was added to the heap multiple times due to bounds tightening
      if leaf is None or leaf.aq_L != lower_bound:
        heapq.heappop(self._all_leaf_lower_bounds)
        continue
      return lower_bound
    if self.leaves:
      # This is an internal-index failure, not a valid empty-tree state.
      raise RuntimeError("Global lower-bound heap is empty while leaves remain")
    return math.inf

  def optimality_tolerance(self, epsilon_abs: float, epsilon_rel: float, lower_bound: float) -> float:
    return float(epsilon_abs) + float(epsilon_rel) * max(
        1.0, abs(self.incumbent_value), abs(lower_bound)
    )

  def gap(self) -> float:
    return self.incumbent_value - self.global_lower_bound()

  def is_certified(self, epsilon_abs: float, epsilon_rel: float) -> bool:
    if not math.isfinite(self.incumbent_value):
      return False
    lower_bound = self.global_lower_bound()
    if not math.isfinite(lower_bound):
      return False
    return self.incumbent_value - lower_bound <= self.optimality_tolerance(
        epsilon_abs, epsilon_rel, lower_bound
    )

  # ------------------------------------------------------------------------
  # Dispatch and result acceptance
  # ------------------------------------------------------------------------
  def _pop_valid_ready_id(self) -> Optional[int]:
    '''
    Find next BnB node to branch on.
    '''
    while self.ready:
      lower_bound, negative_depth, node_id = heapq.heappop(self.ready)
      leaf = self.leaves.get(node_id)
      if leaf is None:
        continue
      if leaf.state != LeafState.READY:
        continue
      if leaf.aq_L != lower_bound or leaf.depth != -negative_depth:
        continue
      return node_id
    return None

  def dispatch_next(self, metadata: Optional[Mapping[str, Any]] = None) -> Optional[BnBNode]:
    '''
    Send the BnB node to the inflight queue and update its status in node partition
    '''
    node_id = self._pop_valid_ready_id()
    if node_id is None:
      return None
    leaf = self.leaves[node_id]
    leaf.state = LeafState.INFLIGHT
    attempt = int(leaf.metadata.get("task_attempt", 0)) + 1
    leaf.metadata["task_attempt"] = attempt
    self.inflight[node_id] = InFlightRecord(
        task_id=node_id,
        parent_id=node_id,
        generation=self.generation,
        submitted_at=time.time(),
        incumbent_at_submit=self.incumbent_value,
        attempt=attempt,
        metadata=dict(metadata or {}),
    )
    return leaf

  def rollback_dispatch(self, parent_id: int, error: Optional[str] = None) -> None:
    record = self.inflight.pop(parent_id, None)
    leaf = self.leaves.get(parent_id)
    if record is None or leaf is None:
      return
    if leaf.state != LeafState.INFLIGHT:
      raise RuntimeError("Only an in-flight leaf can be rolled back")
    leaf.state = LeafState.READY
    if error is not None:
      leaf.metadata.setdefault("task_errors", []).append(str(error))
    self._push_ready(leaf)

  @staticmethod
  def _contains(leaf: BnBNode, x: np.ndarray, tol: float = 1.0e-12) -> bool:
    return bool(np.all(x >= leaf.l - tol) and np.all(x <= leaf.u + tol))

  def _validate_axis_split(self, parent: BnBNode, children: Sequence[BnBNode]) -> None:
    if len(children) != 2:
      raise ValueError("A spatial binary branch must return exactly two children")
    c1, c2 = children
    tol = 1.0e-11 * max(1.0, parent.diam)

    for child in children:
      if child.l.shape != parent.l.shape:
        raise ValueError("Child dimension differs from parent dimension")
      if np.any(child.l < parent.l - tol) or np.any(child.u > parent.u + tol):
        raise ValueError("Child box is not contained in its parent")

    def orientation_ok(left: BnBNode, right: BnBNode) -> bool:
      for split_dim in range(parent.l.size):
        mask = np.ones(parent.l.size, dtype=bool)
        mask[split_dim] = False
        if not np.allclose(left.l, parent.l, atol=tol, rtol=0.0):
          continue
        if not np.allclose(right.u, parent.u, atol=tol, rtol=0.0):
          continue
        if not np.allclose(left.u[mask], parent.u[mask], atol=tol, rtol=0.0):
          continue
        if not np.allclose(right.l[mask], parent.l[mask], atol=tol, rtol=0.0):
          continue
        split_left = left.u[split_dim]
        split_right = right.l[split_dim]
        if not math.isclose(split_left, split_right, abs_tol=tol, rel_tol=0.0):
          continue
        if not (parent.l[split_dim] + tol < split_left < parent.u[split_dim] - tol):
          continue
        return True
      return False

    if not (orientation_ok(c1, c2) or orientation_ok(c2, c1)):
      raise ValueError("Children do not form an axis-aligned binary partition of the parent")

  def _prepare_child(self, parent: BnBNode, child: BnBNode) -> BnBNode:
    child = child.clone()
    child.parent_id = parent.node_id
    child.depth = parent.depth + 1
    child.generation = self.generation
    child.state = LeafState.READY
    child.close_reason = None
    child.parent_aq_L = parent.aq_L
    child.parent_aq_U = parent.aq_U
    child.parent_aq_U_x = (
        None if parent.aq_U_x is None else parent.aq_U_x.copy()
    )
    child_aq_L = child.aq_L;
    child.aq_L = max(child.aq_L, parent.aq_L)

    scale = max(1.0, abs(child.aq_L), abs(child.aq_U))
    mismatch = child.aq_L - child.aq_U
    if mismatch > self.bound_consistency_tol * scale:
      #print("WARNING: Bounds: child L={0:1.16e} parent L={1:1.16e} U={2:1.16e}".format(child.aq_L, parent.aq_L,  child.aq_U), flush=True);
      raise ValueError(
          "Child lower bound exceeds its feasible upper bound after enforcing "
          "parent-child monotonicity: LB=%r, UB=%r" % (child.aq_L, child.aq_U)
      )
    if mismatch > 0.0:
      # Tiny numerical repair only.  Record it explicitly because this is the
      # sole exceptional path that relaxes exact parent-child monotonicity.
      child.metadata["tiny_bound_repair"] = {
          "monotone_lb": child.aq_L,
          "feasible_ub": child.aq_U,
      }
      child.aq_L = child.aq_U

    self._prepare_id(child)
    return child

  def _propagate_parent_feasible_point(
      self, parent: BnBNode, children: Sequence[BnBNode]
  ) -> None:
    if parent.aq_U_x is None or not math.isfinite(parent.aq_U):
      return
    containing = [child for child in children if self._contains(child, parent.aq_U_x)]
    if not containing:
      raise ValueError("Parent feasible point is not contained in either child")
    target = min(containing, key=lambda leaf: int(leaf.node_id))
    if parent.aq_U < target.aq_U:
      target.aq_U = parent.aq_U
      target.aq_U_x = parent.aq_U_x.copy()
      target.metadata["inherited_parent_feasible_point"] = True
      if target.aq_L > target.aq_U:
        scale = max(1.0, abs(target.aq_L), abs(target.aq_U))
        if target.aq_L - target.aq_U <= self.bound_consistency_tol * scale:
          target.aq_L = target.aq_U
        else:
          raise ValueError("Inherited parent feasible point contradicts child lower bound")

  def accept_result(self, result: BranchResult) -> Tuple[Tuple[BnBNode, ...], bool]:
    """
    Atomically replace an in-flight parent by its two returned children. Children may be
    added as READY or CLOSED (pruned or due to small local gap). If the result changes
    the best/least upper bound, additional pruning can occur.

    Returns an empty tuple for a stale/duplicate result.  Worker-reported errors
    are raised after the parent is put back in the ready heap, so callers may
    retry or terminate without losing the parent's regional certificate.
    """
    if result.generation != self.generation:
      self.stale_results += 1
      return tuple()

    parent_id = int(result.parent_id)
    record = self.inflight.get(parent_id)
    parent = self.leaves.get(parent_id)
    if record is None or parent is None or parent.state != LeafState.INFLIGHT:
      self.stale_results += 1
      return tuple()

    if result.error is not None:
      self.rollback_dispatch(parent_id, result.error)
      raise RuntimeError("Worker failed for parent %s: %s" % (parent_id, result.error))

    raw_children = list(result.children)
    self._validate_axis_split(parent, raw_children)
    children = [self._prepare_child(parent, child) for child in raw_children]
    self._propagate_parent_feasible_point(parent, children)

    # Update upper bound before classifying any child, as required by Section 2.3.
    previous_incumbent = self.incumbent_value
    winning_child: Optional[BnBNode] = None
    for child in children:
      if child.aq_U_x is not None and child.aq_U < self.incumbent_value:
        self.incumbent_value = child.aq_U
        self.incumbent_x = child.aq_U_x.copy()
        winning_child = child

    incumbent_changed = False
    if self.incumbent_value < previous_incumbent:
      incumbent_changed = True
      self._reclassify_ready_after_incumbent_update()

    for child in children:
      self._classify(child)

    # Determine which new child owns an incumbent point formerly associated
    # with the parent.  The point and value remain globally valid even if the
    # child workers did not rediscover them.
    replacement_incumbent_leaf_id: Optional[int] = None
    if winning_child is not None:
      replacement_incumbent_leaf_id = winning_child.node_id
    elif self.incumbent_leaf_id == parent_id and self.incumbent_x is not None:
      containing = [child for child in children if self._contains(child, self.incumbent_x)]
      if containing:
        replacement_incumbent_leaf_id = min(
            int(child.node_id) for child in containing if child.node_id is not None
        )

    # The mutation below is the atomic partition replacement.  Everything that
    # can fail was validated/prepared above.
    del self.leaves[parent_id]
    del self.inflight[parent_id]
    for child in children:
      self._insert_leaf(child)
    if replacement_incumbent_leaf_id is not None:
      self.incumbent_leaf_id = replacement_incumbent_leaf_id

    self.accepted_parent_tasks += 1
    if self.debug_checks:
      self.assert_invariants()
    return tuple(children), incumbent_changed

  # ------------------------------------------------------------------------
  # Warm start, views, and diagnostics
  # ------------------------------------------------------------------------
  def restart_from_partition(
      self,
      old_leaves: Iterable[BnBNode],
      transfer_lower_bound: Callable[[BnBNode], float],
      evaluate_upper_point: Callable[[np.ndarray], float],
  ) -> None:
    """Reclassify a retained leaf partition for a new acquisition function.

    ``transfer_lower_bound`` is the explicit hook for the Section-3.5 transfer
    formula.  ``evaluate_upper_point`` reevaluates each stored feasible point.
    """
    if self.inflight:
      raise RuntimeError("Settle or invalidate in-flight work before a warm start")

    old_leaves = list(old_leaves)
    if not old_leaves:
      raise ValueError("Warm-start partition is empty")

    # A fresh BnBAlgorithm object is normally constructed at every BO
    # iteration.  Derive the new generation from the retained records rather
    # than from this new store's zero-valued counter; otherwise two successive
    # BO iterations could both use generation 1 and a late result could collide
    # with a newly dispatched parent carrying the same persistent node ID.
    previous_generation = max(int(leaf.generation) for leaf in old_leaves)
    self.generation = max(self.generation, previous_generation) + 1
    self.leaves.clear()
    self.ready.clear()
    self.inflight.clear()
    self._all_leaf_lower_bounds.clear()
    self.incumbent_value = math.inf
    self.incumbent_x = None
    self.incumbent_leaf_id = None

    prepared: List[BnBNode] = []
    seen_ids = set()
    for old_leaf in old_leaves:
      leaf = old_leaf.clone()
      if leaf.node_id is None:
        self._prepare_id(leaf)
      elif leaf.node_id in seen_ids:
        raise ValueError("Duplicate node ID in warm-start partition")
      else:
        self._prepare_id(leaf)
      seen_ids.add(leaf.node_id)

      leaf.generation = self.generation
      leaf.state = LeafState.READY
      leaf.close_reason = None
      leaf.aq_L = float(transfer_lower_bound(old_leaf))
      if math.isnan(leaf.aq_L):
        raise ValueError("Transferred lower bound is NaN")

      point = leaf.aq_U_x
      if point is None or not self._contains(leaf, point):
        point = leaf.midpoint
      leaf.aq_U_x = np.asarray(point, dtype=float).reshape(-1).copy()
      leaf.aq_U = float(evaluate_upper_point(leaf.aq_U_x))
      if math.isnan(leaf.aq_U):
        raise ValueError("Reevaluated upper bound is NaN")
      if leaf.aq_L > leaf.aq_U:
        scale = max(1.0, abs(leaf.aq_L), abs(leaf.aq_U))
        if leaf.aq_L - leaf.aq_U <= self.bound_consistency_tol * scale:
          leaf.aq_L = leaf.aq_U
        else:
          raise ValueError("Transferred lower bound exceeds reevaluated feasible value")

      prepared.append(leaf)
      if leaf.aq_U < self.incumbent_value:
        self.incumbent_value = leaf.aq_U
        self.incumbent_x = leaf.aq_U_x.copy()
        self.incumbent_leaf_id = leaf.node_id

    # Classify only after the final new incumbent is known.
    for leaf in prepared:
      self._classify(leaf)
      self._insert_leaf(leaf)

    if self.debug_checks:
      self.assert_invariants()

  def export_partition(self) -> List[BnBNode]:
    return [self.leaves[node_id].clone() for node_id in sorted(self.leaves)]

  def ready_nodes(self) -> List[BnBNode]:
    return [leaf for leaf in self.leaves.values() if leaf.state == LeafState.READY]

  def candidate_nodes(self) -> List[BnBNode]:
    """Leaves useful for batching/visualization, excluding only pruned leaves."""
    return [
        leaf for leaf in self.leaves.values()
        if leaf.close_reason != CloseReason.PRUNED.value
    ]

  def incumbent_leaf(self) -> Optional[BnBNode]:
    if self.incumbent_leaf_id in self.leaves:
      return self.leaves[int(self.incumbent_leaf_id)]
    if self.incumbent_x is None:
      return None
    containing = [leaf for leaf in self.leaves.values() if self._contains(leaf, self.incumbent_x)]
    if not containing:
      return None
    leaf = min(containing, key=lambda item: int(item.node_id))
    self.incumbent_leaf_id = leaf.node_id
    return leaf

  def counts(self) -> Dict[str, int]:
    result = {state.value: 0 for state in LeafState}
    for leaf in self.leaves.values():
      result[leaf.state.value] += 1
    result["total"] = len(self.leaves)
    result["pruned"] = sum(
        leaf.close_reason == CloseReason.PRUNED.value for leaf in self.leaves.values()
    )
    result["local_gap"] = sum(
        leaf.close_reason == CloseReason.LOCAL_GAP.value for leaf in self.leaves.values()
    )
    return result

  def assert_invariants(self) -> None:
    if not self.leaves:
      raise AssertionError("The leaf partition must not be empty")

    inflight_ids = {
        node_id for node_id, leaf in self.leaves.items()
        if leaf.state == LeafState.INFLIGHT
    }
    if inflight_ids != set(self.inflight):
      raise AssertionError("InFlight map and leaf states disagree")

    for node_id, leaf in self.leaves.items():
      if leaf.node_id != node_id:
        raise AssertionError("Leaf dictionary key differs from leaf ID")
      if leaf.generation != self.generation:
        raise AssertionError("Leaf belongs to a stale generation")
      scale = max(1.0, abs(leaf.aq_L), abs(leaf.aq_U))
      if leaf.aq_L - leaf.aq_U > self.bound_consistency_tol * scale:
        raise AssertionError("Leaf lower bound exceeds its feasible upper bound")
      if leaf.state == LeafState.CLOSED and leaf.close_reason is None:
        raise AssertionError("Closed leaf has no close reason")
      if leaf.state != LeafState.CLOSED and leaf.close_reason is not None:
        raise AssertionError("Open leaf unexpectedly has a close reason")

    lower_bound = self.global_lower_bound()
    direct_lower_bound = min(leaf.aq_L for leaf in self.leaves.values())
    if lower_bound != direct_lower_bound:
      raise AssertionError("Global lower-bound index is inconsistent")

    if self.incumbent_x is not None:
      if not any(self._contains(leaf, self.incumbent_x) for leaf in self.leaves.values()):
        raise AssertionError("Incumbent point is outside the retained partition")


def initialize_async_search(
    algorithm: Any,
    l0: Optional[np.ndarray] = None,
    u0: Optional[np.ndarray] = None,
    queue: Optional[Sequence[Tuple[float, int, BnBNode]]] = None,
    partition: Optional[Iterable[BnBNode]] = None,
    transfer_lower_bound: Optional[Callable[[BnBNode], float]] = None,
) -> None:
  """Adapter used by ``BnBAlgorithm.initialize``."""
  if l0 is None or u0 is None:
    l_init = np.asarray(algorithm.gpsurrogate.xlimits[:, 0], dtype=float)
    u_init = np.asarray(algorithm.gpsurrogate.xlimits[:, 1], dtype=float)
  else:
    l_init = np.asarray(l0, dtype=float)
    u_init = np.asarray(u0, dtype=float)

  # Defaults keep this adapter compatible with the current branch while the
  # constructor options are migrated.
  if not hasattr(algorithm, "inflight_factor"):
    algorithm.inflight_factor = 1.
  if not hasattr(algorithm, "poll_interval"):
    algorithm.poll_interval = 1.0e-1
  if not hasattr(algorithm, "max_task_retries"):
    algorithm.max_task_retries = 1
  if not hasattr(algorithm, "bound_consistency_tol"):
    algorithm.bound_consistency_tol = 1.0e-4

  store = AsyncLeafPartition(
      epsilon_prune=algorithm.epsilon_prune,
      epsilon_node=algorithm.epsilon_node,
      bound_consistency_tol=algorithm.bound_consistency_tol,
      debug_checks=getattr(algorithm, "enable_debug_checks", False),
  )
  algorithm.leaf_partition = store

  if partition is None:
    aq_L, aq_U, aq_U_x, diagnostics = algorithm.compute_acqf_bounds(l_init, u_init)
    root = BnBNode(l_init, u_init, aq_L, aq_U, aq_U_x=aq_U_x, metadata={"diagnostics" : diagnostics})
    store.initialize_root(root)
    print(f"\nInitial acquisition bounds: lower: {aq_L}   upper: {aq_U}")
    print(f"\nInitial  bounds: lower: {l_init}   upper: {u_init}")
    
    # A legacy queue does not include pruned/closed leaves and therefore is not
    # a spatial partition.  It may safely provide incumbent hints only.
    if queue is not None:
      for _, _, legacy_node in queue:
        point = legacy_node.aq_U_x
        if point is None:
          point = legacy_node.midpoint
        value = float(
            np.asarray(algorithm.acqf.evaluate(np.atleast_2d(point))).reshape(-1)[0]
        )
        if value < store.incumbent_value:
          store.incumbent_value = value
          store.incumbent_x = np.asarray(point, dtype=float).copy()
          store.incumbent_leaf_id = store.incumbent_leaf().node_id
      store._reclassify_ready_after_incumbent_update()
  else:
    old_partition = list(partition)
    if transfer_lower_bound is None:
      # Correct fallback: recompute every lower bound.  The Section-3.5 transfer
      # callback should replace this when available to retain the speed benefit.
      def transfer_lower_bound(old_leaf: BnBNode) -> float:
        lower, _, _, _ = algorithm.compute_acqf_bounds(old_leaf.l, old_leaf.u)
        return float(lower)

    def evaluate_upper_point(point: np.ndarray) -> float:
      return float(
          np.asarray(algorithm.acqf.evaluate(np.atleast_2d(point))).reshape(-1)[0]
      )

    store.restart_from_partition(
        old_partition,
        transfer_lower_bound=transfer_lower_bound,
        evaluate_upper_point=evaluate_upper_point,
    )

  algorithm.LUB = store.incumbent_value
  algorithm.LLB = store.global_lower_bound()
  algorithm.best_node = store.incumbent_leaf()
  algorithm._refresh_legacy_views()


def run_async_search(
    algorithm: Any,
    brancher_type: Any,
    l_init: np.ndarray,
    u_init: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
  """Certified asynchronous master event loop for ``BnBAlgorithm``.

  A bounded pool of brancher objects is used so that no mutable CVXPY/solver
  wrapper is shared by two concurrent thread tasks.  At exit, already submitted
  work is settled and consumed before the GP can be retrained by the BO loop.
  """
  if not hasattr(algorithm, "leaf_partition"):
    initialize_async_search(algorithm, l_init, u_init)
  store: AsyncLeafPartition = algorithm.leaf_partition

  evaluator = algorithm.node_evaluator
  num_workers = max(1, int(evaluator.num_workers()))

  log = algorithm.log

  inflight_limit = max(1, int(math.ceil(algorithm.inflight_factor * num_workers)))
  poll_interval = max(0.0, float(algorithm.poll_interval))

  start_time = time.time()
  algorithm.num_parent_tasks = 0
  # Preserve the current branch's convention: num_branches counts generated
  # child nodes (two for each accepted binary parent task).
  algorithm.num_branches = 0
  algorithm.certified = False
  algorithm.stop_reason = None
  algorithm.drained_inflight_tasks = 0
  algorithm.gap_history = [store.gap()]
  algorithm.branch_history = [0]
  algorithm.prunedvol_history = [
      sum(leaf.volume for leaf in store.leaves.values()
          if leaf.close_reason == CloseReason.PRUNED.value)
  ]

  algorithm.totalvol = sum(leaf.volume for leaf in store.leaves.values())
  log.info("BnB domain total volume: %g" % algorithm.totalvol)
  
  algorithm.pruningratio_history = [
      store.counts()["pruned"] / max(1, store.counts()["total"])
  ]
  algorithm.print_iter_next = 1
  algorithm.print_iter_count = 0

  def print_get_next_target(target):
    if target == 1:
        return 10

    scale = 10 ** (len(str(target)) - 1)
    return target + scale

  def gap_diagnostic(store: Any) -> str:
    return f"GAP: global {store.gap():11.4e}   LLB {store.global_lower_bound():11.4e} LUB={store.incumbent_value:11.4e}"

  def leaf_diagnostic(l: BnBNode, prefix="") -> str:
    output = f"{prefix} id={l.node_id} depth={l.depth} state={l.state} LB={l.aq_L:11.4e} UB={l.aq_U:11.4e} diam={l.diam:11.4e}\n"
    output += f"         {prefix} l        {np.array2string(l.l, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:11.4e}'})}\n"
    output += f"         {prefix} feasib x {np.array2string(l.aq_U_x, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:11.4e}'})}\n"
    output += f"         {prefix} u        {np.array2string(l.u, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:11.4e}'})}"
    return output
    
  def print_iter_info(algorithm: Any, store: Any, log: Any, iter_type: Int) -> None:
    """
    Print a short summary of the search stats
    """
    counts_dict = store.counts()
    if algorithm.print_iter_count % 10 == 0:
      msg = f"# branches  optim gap    PrunedVol(%) PrunedRatio | "
      keys_str = "   ".join(counts_dict.keys())
      msg = f"  {msg}    {keys_str} |     LUB"
      log.info(msg)
    algorithm.print_iter_count += 1

    vals_str = " ".join(f"{v:8d}" for v in counts_dict.values())

    prunedvol_perc = algorithm.prunedvol_history[-1] / algorithm.totalvol * 100.
    msg = ( f"{algorithm.num_branches:8d}  {algorithm.gap_history[-1]:12.5e}  "
            f"{prunedvol_perc:12.5f} {algorithm.pruningratio_history[-1]:12.5e}")

    msg = f"{msg} | {vals_str}"

    if iter_type==1:
      msg = f"* {msg}      | {store.incumbent_value:12.5e}"
    elif iter_type==2:      
      msg = f"f {msg}"
    else:
      msg = f"  {msg}"
      
    log.info(msg)

    if algorithm.diagnostics and iter_type>=1:
      glb_leaf = min(store.leaves.values(), key=lambda leaf: (float(leaf.aq_L), int(leaf.node_id)))
      inc_leaf = store.incumbent_leaf()
      
      log.info(gap_diagnostic(store))
      log.info(leaf_diagnostic(glb_leaf, "LLB    leaf:"))
      log.info(leaf_diagnostic(inc_leaf, "INCUMB leaf:"))
      
      log.info("LLB    leaf conic relax\n" + " "*8 + glb_leaf.metadata["diagnostics"].replace("\n", "\n"+" "*8))
      log.info("INCUMB leaf conic relax\n" + " "*8 + inc_leaf.metadata["diagnostics"].replace("\n", "\n"+" "*8))

  
  def make_brancher() -> Any:
    return brancher_type(
      algorithm.acqf,
      LUB=store.incumbent_value,
      epsilon_prune=algorithm.epsilon_prune,
      acqf_UB_solver=algorithm.acqf_UB_solver,
      random_seed=algorithm.random_seed,
      opt_mode=algorithm.opt_mode,
      nearest_neighbor_pairs=algorithm.nearest_neighbor_pairs,
      diagnostics=algorithm.diagnostics,
    )

  # A brancher owns mutable relaxation/solver objects.  Keep at most one active
  # task per instance; completed instances are reused to bound setup/memory.
  idle_branchers: List[Any] = []
  active_branchers: Dict[int, Any] = {}

  def acquire_brancher(parent_id: int) -> Any:
    brancher = idle_branchers.pop() if idle_branchers else make_brancher()
    if hasattr(brancher, "LUB"):
      brancher.LUB = store.incumbent_value
    if algorithm.random_seed is not None and hasattr(brancher, "rng"):
      # Make stochastic local-UB starts depend on the logical node, not on the
      # nondeterministic order in which brancher instances become idle.
      seed = (
          int(algorithm.random_seed)
          + 1000003 * int(store.generation)
          + 9176 * int(parent_id)
      )
      brancher.random_seed = seed
      brancher.rng = np.random.default_rng(seed)
    active_branchers[parent_id] = brancher
    return brancher

  def release_brancher(parent_id: int) -> None:
    brancher = active_branchers.pop(int(parent_id), None)
    if brancher is not None:
      idle_branchers.append(brancher)

  def record_children(children: Sequence[BnBNode]) -> None:
    if not children:
      return
    algorithm.num_parent_tasks += 1
    algorithm.num_branches += len(children)
    algorithm.branch_history.append(algorithm.num_branches)
    algorithm.gap_history.append(store.gap())
    counts = store.counts()
    algorithm.prunedvol_history.append(
        sum(leaf.volume for leaf in store.leaves.values()
            if leaf.close_reason == CloseReason.PRUNED.value)
    )
    algorithm.pruningratio_history.append(
        counts["pruned"] / max(1, counts["total"])
    )

  def accept_completed_result(result: BranchResult, allow_retry: bool) -> tuple[bool, bool]:
    if result is None:
      raise RuntimeError("The asynchronous evaluator returned a missing result")
    parent_id = int(result.parent_id)
    record = store.inflight.get(parent_id)
    # A late result from an older BO generation can have the same persistent
    # parent ID as a current task.  Never release/reuse the current task's
    # brancher on the basis of that stale result.
    incumbent_changed = False
    if record is not None and int(result.generation) == int(record.generation):
      release_brancher(parent_id)
    try:
      children, incumbent_changed = store.accept_result(result)
    except RuntimeError as exc:
      print(exc)
      parent = store.leaves.get(int(result.parent_id))
      attempts = 0 if parent is None else int(parent.metadata.get("task_attempt", 0))
      if (not allow_retry) or attempts > algorithm.max_task_retries:
        algorithm.last_worker_error = str(exc)
        if algorithm.stop_reason is None:
          algorithm.stop_reason = "worker_failure"
        return False, incumbent_changed
      return True, incumbent_changed
    if children:
      record_children(children)
      return True, incumbent_changed
    return False, incumbent_changed

  while True:
    made_progress = False
    incumbent_changed = False
    completed = evaluator.retrieve_results()
    for result in completed:
      made_progress2, incumbent_changed2 = accept_completed_result(result, allow_retry=True)
      made_progress = made_progress2 or made_progress
      incumbent_changed = incumbent_changed2 or incumbent_changed
      if algorithm.stop_reason == "worker_failure":
        break

    #if len(completed) > 0:
    #  print(f"Evaluator retrieve_results: {len(completed)}  all results were processed.")
      #print(f"BnB nodes/leaves in the partition: {store.counts()}")

    #if incumbent_changed:
    #  print(f"  Counts after : {store.counts()}")
      
    if algorithm.stop_reason == "worker_failure":
      break

    #print(f"[0]Nodes in READY {len(store.ready_nodes())}  in INFLIGHT {len(store.inflight)}  leaves in partition {len(store.leaves)}")

    
    algorithm.LUB = store.incumbent_value
    algorithm.LLB = store.global_lower_bound()
    algorithm.best_node = store.incumbent_leaf()
    algorithm._refresh_legacy_views()


    # print iteration info
    if algorithm.num_branches >= algorithm.print_iter_next or incumbent_changed:
      print_iter_info(algorithm, store, log, int(incumbent_changed))
      #if not incumbent_changed: 
      while algorithm.num_branches >= algorithm.print_iter_next:
        algorithm.print_iter_next = print_get_next_target(algorithm.print_iter_next)
    
        
    if store.is_certified(algorithm.epsilon_gap, algorithm.epsilon_rel_gap):
      algorithm.certified = True
      algorithm.stop_reason = "optimality_gap"
      break
    if algorithm.num_branches >= algorithm.max_bnbiter:
      algorithm.stop_reason = "max_iter"
      break
    if time.time() - start_time >= algorithm.max_bnbtime:
      algorithm.stop_reason = "max_time"
      break

    # Fill only a bounded speculative window.  The parent leaf remains in the
    # partition and retains its certified lower bound while this task runs.
    capacity = inflight_limit - len(store.inflight)
    submitted = 0
    depth_max = 0
    depth_min = 100000
    while capacity > 0:
      parent = store.dispatch_next(
          metadata={"incumbent_at_submit": store.incumbent_value}
      )
      if parent is None:
        break
      depth_min = min(depth_min, parent.depth)
      depth_max = max(depth_max, parent.depth)
      parent_id = int(parent.node_id)
      brancher = acquire_brancher(parent_id)
      try:
        evaluator.submit_tasks(brancher.callback, np.asarray([parent], dtype=object))        
      except Exception as exc:
        release_brancher(parent_id)
        store.rollback_dispatch(parent_id, str(exc))
        raise
      submitted += 1
      capacity -= 1
      made_progress = True

    #if submitted:
    #  print(f"Submitted {submitted} new nodes. Capacity {capacity} out of {inflight_limit}. Sync {algorithm.synchronous}  Min/Max depths {depth_min} <<<< {depth_max}")
      
    if algorithm.synchronous and submitted:
      evaluator.sync()

    if not store.ready_nodes() and not store.inflight:
      # Every leaf is closed.  This should normally imply certification when
      # epsilon_node is no larger than the requested global tolerance.
      algorithm.stop_reason = "all_leaves_closed"
      algorithm.certified = store.is_certified(
          algorithm.epsilon_gap, algorithm.epsilon_rel_gap
      )
      break

    if not made_progress and store.inflight and poll_interval > 0.0:
      time.sleep(poll_interval)

  # A certified lower bound permits the master to stop dispatching immediately,
  # but the current evaluator API has no reliable cancellation operation.  Drain
  # the bounded speculative window before BO mutates/retrains the shared GP.
  # Accepting these results only refines the partition: child LB >= parent LB and
  # any new feasible point can only improve U*.
  if store.inflight:
    inflight_before_drain = len(store.inflight)
    evaluator.sync()
    for result in evaluator.retrieve_results():
      accept_completed_result(result, allow_retry=False)
    algorithm.drained_inflight_tasks = inflight_before_drain
    if store.inflight:
      missing = sorted(store.inflight)
      raise RuntimeError(
          "Evaluator synchronized but did not return results for in-flight "
          "parents %r" % missing
      )

  algorithm.LUB = store.incumbent_value
  algorithm.LLB = store.global_lower_bound()

  algorithm.final_gap = store.gap()
  algorithm.elapsed_bnb_time = time.time() - start_time
  algorithm.best_node = store.incumbent_leaf()
  algorithm._refresh_legacy_views()

  # The certificate can only improve during the exit drain.
  if store.is_certified(algorithm.epsilon_gap, algorithm.epsilon_rel_gap):
    algorithm.certified = True

  best_leaf = store.incumbent_leaf()
  if best_leaf is None:
    # No feasible upper point was returned.  This is valid as an uncertified
    # diagnostic state but optimize() still needs a box to report.
    best_leaf = min(store.leaves.values(), key=lambda leaf: (leaf.aq_L, leaf.node_id))
  algorithm.final_diameter = best_leaf.diam

  if getattr(algorithm, "saveData", False) and algorithm.saveData:

    prefix = str(getattr(algorithm, "saveDataDir", ""))
    suffix = "_BOit" + str(getattr(algorithm, "BOit", 0)) + ".dat"
    np.savetxt(prefix + "branch_history" + suffix, algorithm.branch_history)
    np.savetxt(prefix + "gap_history" + suffix, algorithm.gap_history)
    np.savetxt(prefix + "prunedvol_history" + suffix, algorithm.prunedvol_history)
    np.savetxt(prefix + "pruningratio_history" + suffix, algorithm.pruningratio_history)
    for label, nodes in (
        ("pruned_nodes", algorithm.all_prunednodes),
        ("nonpruned_nodes", algorithm.all_nonpruned_nodes),
    ):
      np.savetxt(prefix + label + "_ls" + suffix, np.asarray([node.l for node in nodes]))
      np.savetxt(prefix + label + "_us" + suffix, np.asarray([node.u for node in nodes]))
      np.savetxt(prefix + label + "_aqU" + suffix, np.asarray([node.aq_U for node in nodes]))
      np.savetxt(prefix + label + "_aqL" + suffix, np.asarray([node.aq_L for node in nodes]))

  incumbent_x = None if store.incumbent_x is None else store.incumbent_x.copy()

  print_iter_info(algorithm, store, log, 2) 
  log.info("BnB finished with status [%s] in %g seconds.", algorithm.stop_reason, time.time() - start_time)
  log.info("BnB returned LLB=%14.8e LUB=%14.8e gap=%14.8e.", algorithm.LLB, algorithm.LUB, algorithm.final_gap)
  return best_leaf.l.copy(), best_leaf.u.copy(), store.incumbent_value, incumbent_x
