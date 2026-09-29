* Source:     Pyomo MPS Writer
* Format:     Free MPS
*
NAME unknown
OBJSENSE
 MIN
ROWS
 N  obj
 E  c_e_eq_pf_branch(_1_)_
 E  c_e_eq_pf_branch(_2_)_
 E  c_e_eq_pf_branch(_3_)_
 E  c_e_eq_p_balance(_1_)_
 E  c_e_eq_p_balance(_2_)_
 E  c_e_eq_p_balance(_3_)_
 G  c_l_ineq_pf_branch_thermal_lb(_1_)_
 G  c_l_ineq_pf_branch_thermal_lb(_2_)_
 G  c_l_ineq_pf_branch_thermal_lb(_3_)_
 L  c_u_ineq_pf_branch_thermal_ub(_1_)_
 L  c_u_ineq_pf_branch_thermal_ub(_2_)_
 L  c_u_ineq_pf_branch_thermal_ub(_3_)_
 L  c_u_pg_piecewise_cost_cons(_1__0)_
 L  c_u_pg_piecewise_cost_cons(_1__1)_
 L  c_u_pg_piecewise_cost_cons(_1__2)_
 L  c_u_pg_piecewise_cost_cons(_1__3)_
 L  c_u_pg_piecewise_cost_cons(_1__4)_
 L  c_u_pg_piecewise_cost_cons(_1__5)_
 L  c_u_pg_piecewise_cost_cons(_1__6)_
 L  c_u_pg_piecewise_cost_cons(_1__7)_
 L  c_u_pg_piecewise_cost_cons(_1__8)_
 L  c_u_pg_piecewise_cost_cons(_1__9)_
 L  c_u_pg_piecewise_cost_cons(_2__0)_
 L  c_u_pg_piecewise_cost_cons(_2__1)_
 L  c_u_pg_piecewise_cost_cons(_2__2)_
 L  c_u_pg_piecewise_cost_cons(_2__3)_
 L  c_u_pg_piecewise_cost_cons(_2__4)_
 L  c_u_pg_piecewise_cost_cons(_2__5)_
 L  c_u_pg_piecewise_cost_cons(_2__6)_
 L  c_u_pg_piecewise_cost_cons(_2__7)_
 L  c_u_pg_piecewise_cost_cons(_2__8)_
 L  c_u_pg_piecewise_cost_cons(_2__9)_
 E  c_e_pg_piecewise_cost_cons(_3__0)_
COLUMNS
     va(_2_) c_e_eq_pf_branch(_2_)_ -1.3333333333333333
     va(_2_) c_e_eq_pf_branch(_3_)_ -1.1111111111111112
     va(_3_) c_e_eq_pf_branch(_1_)_ -1.6129032258064517
     va(_3_) c_e_eq_pf_branch(_2_)_ 1.3333333333333333
     pg(_1_) c_e_eq_p_balance(_1_)_ 1
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__0)_ 2700
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__1)_ 7100
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__2)_ 11500
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__3)_ 15900
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__4)_ 20300
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__5)_ 24700
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__6)_ 29100
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__7)_ 33500
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__8)_ 37900
     pg(_1_) c_u_pg_piecewise_cost_cons(_1__9)_ 42300
     pg(_2_) c_e_eq_p_balance(_2_)_ 1
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__0)_ 1820.0000000000002
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__1)_ 5220.0000000000009
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__2)_ 8620
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__3)_ 12020.000000000002
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__4)_ 15419.999999999996
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__5)_ 18820.000000000007
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__6)_ 22219.999999999993
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__7)_ 25620.000000000015
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__8)_ 29019.999999999985
     pg(_2_) c_u_pg_piecewise_cost_cons(_2__9)_ 32420
     pg(_3_) c_e_eq_p_balance(_3_)_ 1
     pf(_1_) c_e_eq_pf_branch(_1_)_ 1
     pf(_1_) c_e_eq_p_balance(_1_)_ -1
     pf(_1_) c_e_eq_p_balance(_3_)_ 1
     pf(_1_) c_l_ineq_pf_branch_thermal_lb(_1_)_ 1
     pf(_1_) c_u_ineq_pf_branch_thermal_ub(_1_)_ 1
     pf(_2_) c_e_eq_pf_branch(_2_)_ 1
     pf(_2_) c_e_eq_p_balance(_2_)_ 1
     pf(_2_) c_e_eq_p_balance(_3_)_ -1
     pf(_2_) c_l_ineq_pf_branch_thermal_lb(_2_)_ 1
     pf(_2_) c_u_ineq_pf_branch_thermal_ub(_2_)_ 1
     pf(_3_) c_e_eq_pf_branch(_3_)_ 1
     pf(_3_) c_e_eq_p_balance(_1_)_ -1
     pf(_3_) c_e_eq_p_balance(_2_)_ 1
     pf(_3_) c_l_ineq_pf_branch_thermal_lb(_3_)_ 1
     pf(_3_) c_u_ineq_pf_branch_thermal_ub(_3_)_ 1
     pg_cost(_1_) obj 1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__0)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__1)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__2)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__3)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__4)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__5)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__6)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__7)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__8)_ -1
     pg_cost(_1_) c_u_pg_piecewise_cost_cons(_1__9)_ -1
     pg_cost(_2_) obj 1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__0)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__1)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__2)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__3)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__4)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__5)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__6)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__7)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__8)_ -1
     pg_cost(_2_) c_u_pg_piecewise_cost_cons(_2__9)_ -1
     pg_cost(_3_) obj 1
     pg_cost(_3_) c_e_pg_piecewise_cost_cons(_3__0)_ 1
RHS
     RHS c_e_eq_pf_branch(_1_)_ 0
     RHS c_e_eq_pf_branch(_2_)_ 0
     RHS c_e_eq_pf_branch(_3_)_ 0
     RHS c_e_eq_p_balance(_1_)_ 1.1000000000000001
     RHS c_e_eq_p_balance(_2_)_ 1.1000000000000001
     RHS c_e_eq_p_balance(_3_)_ 0.94999999999999996
     RHS c_l_ineq_pf_branch_thermal_lb(_1_)_ -90
     RHS c_l_ineq_pf_branch_thermal_lb(_2_)_ -0.5
     RHS c_l_ineq_pf_branch_thermal_lb(_3_)_ -90
     RHS c_u_ineq_pf_branch_thermal_ub(_1_)_ 90
     RHS c_u_ineq_pf_branch_thermal_ub(_2_)_ 0.5
     RHS c_u_ineq_pf_branch_thermal_ub(_3_)_ 90
     RHS c_u_pg_piecewise_cost_cons(_1__0)_ 0
     RHS c_u_pg_piecewise_cost_cons(_1__1)_ 8800
     RHS c_u_pg_piecewise_cost_cons(_1__2)_ 26400
     RHS c_u_pg_piecewise_cost_cons(_1__3)_ 52800
     RHS c_u_pg_piecewise_cost_cons(_1__4)_ 88000
     RHS c_u_pg_piecewise_cost_cons(_1__5)_ 132000
     RHS c_u_pg_piecewise_cost_cons(_1__6)_ 184800
     RHS c_u_pg_piecewise_cost_cons(_1__7)_ 246400
     RHS c_u_pg_piecewise_cost_cons(_1__8)_ 316800
     RHS c_u_pg_piecewise_cost_cons(_1__9)_ 396000
     RHS c_u_pg_piecewise_cost_cons(_2__0)_ 0
     RHS c_u_pg_piecewise_cost_cons(_2__1)_ 6800.0000000000018
     RHS c_u_pg_piecewise_cost_cons(_2__2)_ 20399.999999999996
     RHS c_u_pg_piecewise_cost_cons(_2__3)_ 40800.000000000007
     RHS c_u_pg_piecewise_cost_cons(_2__4)_ 67999.999999999971
     RHS c_u_pg_piecewise_cost_cons(_2__5)_ 102000.00000000007
     RHS c_u_pg_piecewise_cost_cons(_2__6)_ 142799.99999999988
     RHS c_u_pg_piecewise_cost_cons(_2__7)_ 190400.0000000002
     RHS c_u_pg_piecewise_cost_cons(_2__8)_ 244799.99999999977
     RHS c_u_pg_piecewise_cost_cons(_2__9)_ 306000
     RHS c_e_pg_piecewise_cost_cons(_3__0)_ 0
BOUNDS
 LO BOUND va(_2_) -3.1415926535897931
 UP BOUND va(_2_) 3.1415926535897931
 LO BOUND va(_3_) -3.1415926535897931
 UP BOUND va(_3_) 3.1415926535897931
 LO BOUND pg(_1_) 0
 UP BOUND pg(_1_) 20
 LO BOUND pg(_2_) 0
 UP BOUND pg(_2_) 20
 LO BOUND pg(_3_) 0
 UP BOUND pg(_3_) 0
 LO BOUND pf(_1_) -90
 UP BOUND pf(_1_) 90
 LO BOUND pf(_2_) -0.5
 UP BOUND pf(_2_) 0.5
 LO BOUND pf(_3_) -90
 UP BOUND pf(_3_) 90
 FR BOUND pg_cost(_1_)
 FR BOUND pg_cost(_2_)
 FR BOUND pg_cost(_3_)
ENDATA
