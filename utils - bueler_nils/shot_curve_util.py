import numpy as np
from scipy import integrate as si

class fillingCurvePoints:
    # Initializes the physical parameters of the shot sleeve and die.
    def __init__(self, d_M: float = 80e-3, l_m_activ: float = 756e-3, m_total: float = 3.83, m_part_overflow: float = 2.82, \
                 m_overflow: float = 0.12, t_biscuit_solid: float = 55e-3, rho_solid: float = 2.63e3, f_liq_sol: float = 0.92) -> None:
        
        # total metal volume liquid [m^3]
        V_I_liq = m_total/(rho_solid*f_liq_sol)
        # total metal volume solid [m^3]
        V_I_sol = m_total/rho_solid
        # shot sleeve cross section [m^2]
        A_dM = np.square(d_M)*np.pi/4
        # volume biscuit solid [m^3]
        V_biscuit_sol = t_biscuit_solid*A_dM
        # volume biscuit liquid [m^3]
        V_biscuit_liq = V_biscuit_sol/f_liq_sol
        # mold volume [m^3]
        V_mold = V_I_sol - V_biscuit_sol
        # part plus overflow volume solid [m^3]
        V_A_sol = m_part_overflow/rho_solid
        # part without overflow volume solid [m^3]
        V_part_sol = (m_part_overflow - m_overflow)/rho_solid
        # runner volume mold [m^3]
        V_runner = V_mold - V_A_sol
        # shot sleeve volume [m^3]
        V_chamber = A_dM*l_m_activ

        # filling ratio shot sleeve liquid
        self.fr = V_I_liq/V_chamber
        # chamber filled
        self.s_m_100 = l_m_activ*(1-self.fr)
        # runner filled
        self.s_ma = self.s_m_100 + V_runner/A_dM
        # part filled = overflow reached
        s_ov = self.s_ma + V_part_sol/A_dM
        # all filled - still liquid
        self.s_ffin = self.s_ma + V_A_sol/A_dM
        # all filled - solid
        s_Ifin_solid = self.s_ffin + (V_I_liq - V_I_sol)/A_dM

        # additional members from parameters
        self.l_m_activ = l_m_activ
        self.d_M = d_M

        # Set numerical spatial discretization step (0.1 mm default)
        self.cell_size = 1e-4
        # Calculate total number of cells needed for the active length
        nr_of_cells = int(l_m_activ/self.cell_size)+1
        # Create the spatial array (s) from 0 to active length
        self.s = np.linspace(0,l_m_activ,nr_of_cells)

        # Tolerance parameter for Ramer-Douglas-Peucker (RDP) algorithm
        self.epsilon = 5e-3
        # Ensure a non-zero initial velocity to prevent divide-by-zero in time calculation later
        self.v_init = 0.02

    def set_numerical_options(self, cell_size: float = None, v_init: float = None, epsilon: float = None) -> None:
        """Set numerical options. None keeps current value.

        Args:
            cell_size (float, optional): Discretization of shot sleeve. Defaults to 1e-4 [m].
            v_init (float, optional): Initial velocity (positive). Defaults to 2e-2 [m/s].
            epsilon (float, optional): Tolerance of RDP-resampling algorithm. Defaults to 5e-3.
        """
        # Update spatial discretization if provided
        if cell_size is not None:
            self.cell_size = cell_size
            nr_of_cells = int(self.l_m_activ/self.cell_size)+1
            self.s = np.linspace(0,self.l_m_activ,nr_of_cells)
            
        # Update initial velocity, ensuring it remains strictly positive
        if v_init is not None:
            if v_init > 0:
                self.v_init = v_init
            else:
                raise ValueError("v_init needs to be positive. Zero is numerically not admissible.")
                
        # Update RDP simplification tolerance
        if epsilon is not None:
            self.epsilon = epsilon

    def curve_points_time_velocity(self, curve_type: str, *parameters) -> tuple[np.ndarray, np.ndarray]:
        """Generate shot curve points based on the specified curve type and parameters.

        Args:
            curve_type (str): First phase curve type ("Nogowizin", "Buhler any", "No optimization").
            parameters (tuple): Parameters for the selected curve type. 
                - For "Nogowizin": (s3, v3, s4, v4, sbrake, vbrake)
                - For "Buhler": (v_crit, s3, v3, s4, v4, sbrake, vbrake)
                - For "No optimization": (s1, v1, s2, v2, s3, v3, s4, v4, sbrake, vbrake)

        Returns:
            tuple[np.ndarray, np.ndarray]: t, v arrays after resampling.
        """
        if curve_type.lower() == "nogowizin":
            # check input unit
            if np.any(np.array(parameters[::2]) > self.l_m_activ):
                # inconsistent, check if mm was used: divide by 1000 and check if smaller + divide by 100 and check if larger
                if np.any(np.array(parameters[::2])*1e-2 > self.l_m_activ) \
                    and not np.any(np.array(parameters[::2])*1e-3 > self.l_m_activ):
                    # convert to mm and warn
                    parameters = np.array(parameters)
                    parameters[::2] *= 1e-3
                    parameters = tuple(parameters)
                    print("Stroke larger than l_m_activ, reduced stroke values to mm")
                else:
                    # kill
                    raise ValueError("Stroke larger than l_m_activ and not reasonable in mm. Provide stroke in m.")
            # Get first phase analytical curves based on Nogowizin
            s, v, t, s_kr, v_kr, _, _ = self._nogowizin(self.s, self.d_M, self.l_m_activ, self.fr, v_initial=self.v_init)
            # Stitch on the second (fast shot) and braking phases
            s, v, t = self._second_phase(s, v, True, s_kr, v_kr, *parameters)

            # resample based on s, v data to reduce points
            points = np.column_stack((s, v))
            simplified_points = self._rdp(points, epsilon=self.epsilon)  # Adjust epsilon for desired tolerance

            s_simp, v_simp = simplified_points[:, 0], simplified_points[:, 1]
            
            # POTENTIAL ISSUE: Finding time by index mapping.
            # Using _find_nearest_idx forces the time to snap to the nearest original grid point.
            # This causes temporal jitter because the spatial point was simplified by RDP.
            # Requires that cell size is sufficiently small, which is usually given.
            t_simp = np.array([t[self._find_nearest_idx(s, s_simp_i)] for s_simp_i in s_simp])
            
        elif "buhler" in curve_type.lower() or "buehler" in curve_type.lower():
            # check input unit
            if np.any(np.array(parameters[1::2]) > self.l_m_activ):
                # inconsistent, check if mm was used: divide by 1000 and check if smaller + divide by 100 and check if larger
                if np.any(np.array(parameters[1::2])*1e-2 > self.l_m_activ) \
                    and not np.any(np.array(parameters[1::2])*1e-3 > self.l_m_activ):
                    # convert to mm and warn
                    parameters = np.array(parameters)
                    parameters[1::2] *= 1e-3
                    parameters = tuple(parameters)
                    print("Stroke larger than l_m_activ, reduced stroke values to mm")
                else:
                    # kill
                    raise ValueError("Stroke larger than l_m_activ and not reasonable in mm. Provide stroke in m.")
            v_crit = parameters[0]
            # Get first phase analytical curves based on Buhler
            s, v, t, s_kr, _, _, _ = self._buhler(self.s, self.d_M, self.fr, self.s_m_100, v_crit, v_initial=self.v_init)
            # Stitch on the second phase
            s, v, t = self._second_phase(s, v, True, s_kr, v_crit, *parameters[1:])
            
            # resample based on s, v data to reduce points
            points = np.column_stack((s, v))
            simplified_points = self._rdp(points, epsilon=self.epsilon)  # Adjust epsilon for desired tolerance

            s_simp, v_simp = simplified_points[:, 0], simplified_points[:, 1]
            # POTENTIAL ISSUE: Finding time by index mapping.
            # Using _find_nearest_idx forces the time to snap to the nearest original grid point.
            # This causes temporal jitter because the spatial point was simplified by RDP.
            # Requires that cell size is sufficiently small, which is usually given.
            t_simp = np.array([t[self._find_nearest_idx(s, s_simp_i)] for s_simp_i in s_simp])
            
        elif curve_type.lower() == "no optimization":
            # check input unit
            if np.any(np.array(parameters[::2]) > self.l_m_activ):
                # inconsistent, check if mm was used: divide by 1000 and check if smaller + divide by 100 and check if larger
                if np.any(np.array(parameters[::2])*1e-2 > self.l_m_activ) \
                    and not np.any(np.array(parameters[::2])*1e-3 > self.l_m_activ):
                    # convert to mm and warn
                    parameters = np.array(parameters)
                    parameters[::2] *= 1e-3
                    parameters = tuple(parameters)
                    print("Stroke larger than l_m_activ, reduced stroke values to mm")
                else:
                    # kill
                    raise ValueError("Stroke larger than l_m_activ and not reasonable in mm. Provide stroke in m.")
            s_vi1, v_i1, s_vi2, v_i2 = parameters[:4]
            # Generate unoptimized standard curve
            s, v, t = self._standard_curve(self.s, s_vi1, v_i1, s_vi2, v_i2, v_initial=self.v_init)
            # Stitch on the second phase
            s, v, t = self._second_phase(s, v, False, *parameters[2:])
            
            # resample based on t, v data (Notice it uses t, v here instead of s, v)
            points = np.column_stack((t, v))
            simplified_points = self._rdp(points, epsilon=self.epsilon)  # Adjust epsilon for desired tolerance

            t_simp, v_simp = simplified_points[:, 0], simplified_points[:, 1]
        else:
            raise ValueError(f"Unknown curve type '{curve_type}'. Use 'Nogowizin', 'Buhler', or 'no optimization'.")
        
        return t_simp, v_simp
    
    def curve_points_stroke_velocity(self, curve_type: str, *parameters) -> tuple[np.ndarray, np.ndarray]:
        """Generate shot curve points based on the specified curve type and parameters.

        Args:
            curve_type (str): First phase curve type ("Nogowizin", "Buhler any", "No optimization").
            parameters (tuple): Parameters for the selected curve type. 
                - For "Nogowizin": (s3, v3, s4, v4, sbrake, vbrake)
                - For "Buhler": (v_crit, s3, v3, s4, v4, sbrake, vbrake)
                - For "No optimization": (s1, v1, s2, v2, s3, v3, s4, v4, sbrake, vbrake)

        Returns:
            tuple[np.ndarray, np.ndarray]: s, v arrays after resampling.
        """
        t, v = self.curve_points_time_velocity(curve_type, *parameters)
        
        # integrate the velocity-time curve with trapez-rule
        s = si.cumulative_trapezoid(v, t, initial=0)
        
        return s, v

    def _find_nearest_idx(self, array, value):
        # Helper function to find the index of the closest value in an array
        array = np.atleast_1d(array)
        idx = (np.abs(array - value)).argmin()
        return idx
        
    def _get_time_points_discretized(self, s, v):  
        # Initializes time array with zeros 
        t = np.copy(s) * 0.  # Compute time from velocity and position
        
        # POTENTIAL ISSUE: numerics -> requires v_init to be positive, otherwise too slow
        # v_init = 0.02 empirically good value
        # Uses trapezoidal approximation: dt = 2 * ds / (v_current + v_prev)
        dt = (2 * (s - np.roll(s, 1)) / (v + np.roll(v,1)))[1:]    
        
        # Accumulate dt to get absolute time t
        t[1:] = np.cumsum(dt)
        return t
        
    def _get_acceleration(self, v,t):
        # Simple finite difference gradient for acceleration
        return np.gradient(v, t)
        
    def _nogowizin(self, s, dm, lmactiv, fr, v_initial=0):
        # Calculate critical wave velocity
        v_kr=(9.81*dm)**0.5*(1.386-1.915*fr+0.616*fr**2)  # Eq. 5.13 Nogowizin
        # Calculate critical acceleration
        a_kr=9.81*(dm/lmactiv)*(1+fr)/(1-fr)*(0.98-1.354*fr+0.436*fr**2)**2  # Eq. 5.18 Nogowizin
        # Calculate critical stroke
        s_kr=lmactiv*(1-fr)/(1+fr)  # Eq. 5.16 Nogowizin
        
        # Calculate velocity profile assuming purely constant acceleration
        v = np.sqrt(np.square(v_initial) + s*2*a_kr)  # Array for piston velocity, Equation holds for constant acceleration
        
        # Cap velocity at the critical wave velocity
        v[v>v_kr] = v_kr
            
        t = self._get_time_points_discretized(s, v)
        t_kr = t[self._find_nearest_idx(s, s_kr)]
        
        return s, v, t, s_kr, v_kr, a_kr, t_kr
        
    def _buhler(self, s, dm, fr, s_m_100, vc, v_initial=0):
        # Commented out original calculation, overriding with direct vc input
        # v_kr=(9.81*dm)**0.5*(1.386-1.915*fr+0.616*fr**2) * fraction_v_crit  # Eq. 5.13 Nogowizin
        v_kr = vc
        s_kr=s_m_100
        
        # Constant acceleration derived from final required velocity and distance
        a_kr = np.square(v_kr) / (2*s_kr)  # Bühler: v_kr at sm100
            
        # Velocity profile calculation
        v = np.sqrt(np.square(v_initial) + s*2*a_kr)  # Array for piston velocity
        v[v>v_kr] = v_kr
        
        t = self._get_time_points_discretized(s, v)          
        t_kr = t[self._find_nearest_idx(s, s_kr)]
            
        return s, v, t, s_kr, v_kr, a_kr, t_kr
        
    def _standard_curve(self, s, s_vi1, v_i1, s_vi2, v_i2, v_initial=0):
        # Linear interpolation for Sector 0...1   
        dv_ds_1 = (v_i1 - v_initial) / s_vi1
        v = v_initial + dv_ds_1 * s
        v[v>v_i1] = v_i1
        
        # Linear interpolation for Sector 1...2
        if s_vi2 <= s_vi1:
            print("s2 <= s1 is impossible.")
            pass
        else:
            dv_ds_2 = (v_i2 - v_i1) / (s_vi2 - s_vi1)
            v[s>s_vi1] = v_i1 + dv_ds_2 * (s[s>s_vi1]-s_vi1)
            v[v>v_i2] = v_i2
        
        t = self._get_time_points_discretized(s, v)
        return s, v, t
    
    def _point_line_distance(self, point, start, end):
        """Calculate the distance from a point to a line segment."""
        # Standard geometric cross-product distance formula
        if np.all(start == end):
            return np.linalg.norm(point - start)
        else:
            return np.abs(np.linalg.norm(np.cross(end-start, start-point)) / np.linalg.norm(end-start))

    def _rdp(self, points, epsilon):
        """Recursive Ramer-Douglas-Peucker simplification."""
        dmax = 0.0
        index = 0
        start, end = points[0], points[-1]
        
        # Find the point with the maximum distance from the line segment
        for i in range(1, len(points)-1):
            d = self._point_line_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            rec_results1 = self._rdp(points[:index+1], epsilon)
            rec_results2 = self._rdp(points[index:], epsilon)
            result = np.vstack((rec_results1[:-1], rec_results2))
        else:
            result = np.vstack((start, end))
        return result
    
    def _second_phase(self, s: np.ndarray, v: np.ndarray, opt: bool, s2: float, v2: float, s3: float, v3: float, s4: float, 
                     v4: float, sbrake: float, vbrake: float):
        # Finds the indices in the predefined 's' array closest to the target parameters
        index_s_vi3 = self._find_nearest_idx(s, s3)
        index_s_vi4 = self._find_nearest_idx(s, s4)
        index_s_break = self._find_nearest_idx(s, sbrake)
        index_sffin = self._find_nearest_idx(s, self.s_ffin)

        # Apply specific logic based on whether phase 1 was optimized
        if opt:  # With first phase optimization
            index_s_vi25 = self._find_nearest_idx(s, s2)
            v_i25 = v[index_s_vi25]
            # Linearly interpolate velocity
            v[index_s_vi25:index_s_vi3] = v_i25 + (v3 - v_i25) / (s3 - s2) * (s[index_s_vi25:index_s_vi3]-s2)   
        else:  # Without first phase optimization
            index_s_vi2 = self._find_nearest_idx(s, s2)
            # Linearly interpolate velocity
            v[index_s_vi2:index_s_vi3] = v2 + (v3 - v2) / (s3 - s2) * (s[index_s_vi2:index_s_vi3]-s2)

        # Fast shot acceleration phase interpolation
        v[index_s_vi3:index_s_vi4] = v3 + (v4 - v3) / (s4 - s3) * (s[index_s_vi3:index_s_vi4]-s3)
        v[index_s_vi3] = v3

        # Clamp max velocity
        v[v>v4] = v4
        v[index_s_vi4:] = v4

        # Brake phase interpolation
        v[index_s_break:index_sffin] = v4 + (vbrake - v4) / (self.s_ffin - sbrake) * (s[index_s_break:index_sffin]-sbrake)
        v[index_s_break] = v4
        v[index_sffin:] = vbrake

        # Re-calculate discretized time with the newly edited velocity array
        t = self._get_time_points_discretized(s, v)

        return s, v, t