include("calibration_cylinder.jl")
using .CylinderCalibration

using Manifolds, LinearAlgebra, LieGroups
using Enzyme
using StaticArrays

@info "Loading the data..."
pcs,scan_coords = CylinderCalibration.load_2d_scans("scans10");
T_EB_data,x_data,shuffled_pcs = CylinderCalibration.prepare_data(pcs,scan_coords; threshold=1.0);


SE3 = SpecialEuclideanGroup(3);
se3 = LieAlgebra(SE3);
SO3 = SpecialOrthogonalGroup(3);
so3 = LieAlgebra(SO3);
G = GroupAction(LeftMultiplicationGroupAction(), SE3, Euclidean(3))

# Initialize pose
T_CE = Matrix(1e0*I(4));
T_CE[1:3,4] = [200,200,200];
# Initialize cylinder
v = normalize([1,0,0])
c  = apply(G, T_EB_data[1], apply(G, T_CE, x_data[1][end]+[0,0,100])) # place the center on the right side to avoid local minima
r = 25.0 / 2
cylinder = vcat(v,c,r)

# Visualize initial initialization
@info "Preparing visualization"
CylinderCalibration.visualize_calibration_ex(T_EB_data, x_data, T_CE, cylinder, shuffled_pcs)

# Calibrate
data = (T_EB_data,x_data,cylinder,T_CE);
@info "Setting up the solver..."
opt_T_CE, opt_cylinder = CylinderCalibration.solve(data; max_iters=50)

# Visualize calibration results
@info "Showing calibration results"
println("Optimal camera pose:")
display(opt_T_CE)
println("Fitted cylinder:")
display(opt_cylinder)
CylinderCalibration.visualize_calibration_ex(T_EB_data, x_data, opt_T_CE, opt_cylinder,shuffled_pcs)
