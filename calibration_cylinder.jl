module CylinderCalibration

using CSV, DelimitedFiles, StatsBase, ProgressBars
using StaticArrays, LinearAlgebra, LieGroups, Random, Manifolds
Random.seed!(123)
using ForwardDiff, NonlinearSolve, PreallocationTools
using GLMakie
import Rotations as Rot

function visualize_calibration_ex(T_EB_data, x_data, T_CE, cylinder, point_clouds)
    fig = Figure(size = (1200, 800))
    ax = LScene(fig[1, 1], show_axis = false)

    SE3 = SpecialEuclideanGroup(3)
    se3 = LieAlgebra(SE3)
    G = GroupAction(LeftMultiplicationGroupAction(), SE3, Euclidean(3))

    # Plot point clouds
    projected_point_cloud_points = Vector{SVector{3, Float64}}(undef,0)
    for i in 1:length(point_clouds)
        point_cloud = point_clouds[i]
        T_EB = T_EB_data[i]
        append!(projected_point_cloud_points, map(point->apply(G, T_EB, apply(G, T_CE, point)), point_cloud))
    end
    meshscatter!(ax, projected_point_cloud_points, color = :blue, markersize = 1.0, label = "Scans")

    # Plot ellipses
    ellipses = SVector{3, Float64}[]
    for i in 1:length(T_EB_data)
        T_EB = T_EB_data[i]
        points = map(x_data[i]) do x
            apply(G, T_EB, apply(G, T_CE, x))
        end
        append!(ellipses, points)
    end
    meshscatter!(ax, ellipses, color = :red, markersize = 1.5, label = "Ellipses")

    meshscatter!(ax, [SVector{3}(cylinder[4:6])], color = :blue, markersize = 4, label = "c")

    if !isdir("out")
        mkdir("out")
    end
    display(fig)
    cam = cameracontrols(ax.scene)
    zoom!(ax.scene, cam, 0.3)
    translate_cam!(ax.scene, cam, (-200,200,0))
    rotate_cam!(ax.scene, cam, (0,-pi/1.5,0))
    save("out/visualization.png",fig,update=false)
    return nothing
end

function load_2d_scans(root_folder)
    function convert_coordinates(scan)
        PIXEL2MM = 1.125
        BINNING = 5

        z_not_nan = map(!isnan,scan)
        z = 980 .- scan
        N = size(scan,1)
        x = zeros(N)
        lower_bound = -floor(N/2)
        y = (lower_bound+N-1:-1:lower_bound) .* PIXEL2MM / BINNING
        SVector{3,Float64}.(x[z_not_nan],y[z_not_nan],z[z_not_nan])
    end
    calib_dir = joinpath(root_folder,"calibration")
    scan_coords = readdlm(joinpath(calib_dir,"positions.csv"),',')
    output_files = [name for name in readdir(calib_dir) if startswith(name,"output_")]
    sorted_output_files = sort(output_files, by=x->parse(Int64,only(match(r"(\d+)",x))))

    pcs = []
    for file in sorted_output_files
        batch = stack(CSV.read(joinpath(calib_dir,file),Tuple;header=false))
        batch = coalesce.(batch,NaN) #missing -> NaN
        for pc in eachrow(batch)
            push!(pcs,convert_coordinates(pc))
        end
    end
    return pcs, scan_coords
end

function normalize_data(data::AbstractVector{SVector{D,T}}) where {D,T}
    center = mean(data)
    centered_data = data .- Ref(center)
    scale = inv(maximum(norm.(centered_data)))
    return center, scale, scale*centered_data
end

function fit_orthogonal_ellipse(points_2d::AbstractVector{SVector{D,T}}) where {D,T}
    # Normalize data
    center, scale, normalized_data = normalize_data(points_2d)
    u = try
        _fit_orthogonal_ellipse(normalized_data)
    catch e
        rethrow(e)
    end
    M,c = try
        coef_to_matrix(u)
    catch e
        rethrow(e)
    end
    # Rescale
    scaled_M,scaled_c = M*scale*scale,c/scale
    # Recenter
    translated_c = scaled_c + center
    return scaled_M, translated_c
end

function design_matrix(points_2d)
    DDt = @MMatrix zeros(5,5)
    N = size(points_2d,1)
    for i in eachindex(points_2d)
        x,y = points_2d[i]
        full_p = SVector{5}(x^2,y^2,x,y,1)
        DDt .+= full_p*transpose(full_p)/N
    end
    return SMatrix{5,5}(DDt)
end

function _fit_orthogonal_ellipse(points_2d::AbstractVector{SVector{D,P}}) where {D,P}
    DDt = design_matrix(points_2d)
    S1 = SMatrix{2,2}(view(DDt,1:2,1:2))
    S2 = SMatrix{2,3}(view(DDt,1:2,3:5))
    S3 = SMatrix{3,3}(view(DDt,3:5,3:5))

    # CASE λ = 0:
    # Search in the kernel first
    kernel = nullspace(DDt,atol=1e-4)
    if length(kernel) > 0
        for col in eachcol(kernel)
            d = col[1]*col[2]
            if d > 0
                u = col ./ d
                return SVector{6}(u[1],0,u[2],u[3],u[4],u[5])
            end
        end
        throw(DomainError("Degenerate points"))
    end

    # CASE λ != 0:
    # We can invert S3
    F = cholesky(Symmetric(S3))
    T = F \ transpose(S2)
    M = Symmetric(S1 - S2*T)
    M_tilde = SMatrix{2,2}(M[1,1],M[2,1],M[1,2],M[2,2])
    eig_vals, eig_vecs = eigen(M_tilde)
    
    # Find minimum positive eigenvalue with positive determinant
    determinants = SVector{2}(eig_vecs[1,1]*eig_vecs[2,1], eig_vecs[1,2]*eig_vecs[2,2])
    min_idx = 0
    min_cost = Inf
    for i in 1:2
        d = determinants[i]
        v = eig_vals[i]
        if d > 0
            if v < min_cost
                min_idx = i
                min_cost = v
            end
        end
    end
    # Normalize such that a*c = 1
    a1 = 1/sqrt(determinants[min_idx]) * @view(eig_vecs[:,min_idx])
    a2 = -T * a1
    return SVector{6}(a1[1],0,a1[2],a2[1],a2[2],a2[3])
end

function coef_to_matrix(A,B,C,D,E,F)
    M = SMatrix{2,2}(A,B/2,B/2,C)
    b = -SVector{2}(D,E)/2
    x0 = M \ b
    norm_factor = transpose(x0) * (M * x0) - F
    final_M = M/norm_factor
    final_c = x0
    return final_M, final_c
end
coef_to_matrix(u::AbstractVector) = coef_to_matrix(u...)

function ellipseResidual(x, M , x0)
    d = x-x0
    stretch = sqrt(transpose(d)*M*d) - 1
    residual = norm(d) * abs(stretch)/(1+stretch)
    return residual
end

function orthogonal_ransac_ellipse(points_2d;threshold=1.0,max_iters=6000, max_radius=20, min_radius=10,min_samples=4)
    I = axes(points_2d,1)
    N = size(points_2d,1)
    best_score = 0
    best_u = nothing
    best_samples = nothing
    residuals = zeros(size(points_2d,1))
    PIXEL2MM = 1.125
    half_window_length = Int(cld(max_radius,PIXEL2MM))

    for i in 1:max_iters
        # Sample 1 seed point
        seed = rand(I)
        # sample 4 points near the seed
        lb = max(seed-half_window_length,1)
        ub = min(seed+half_window_length,N)
        sampled_idxs = sample(intersect(lb:ub,I),min_samples;replace=false, ordered=true)
        M,c = try
            fit_orthogonal_ellipse(points_2d[sampled_idxs])
        catch e
            continue
        end
        # Pleasible radius
        (sqrt(1/M[1,1]) > max_radius || sqrt(1/M[2,2]) > max_radius) && continue
        (sqrt(1/M[1,1]) < min_radius || sqrt(1/M[2,2]) < min_radius) && continue
        # Residuals and inliers
        residuals .= ellipseResidual.(points_2d,Ref(M),Ref(c))
        inliers = @view I[residuals .<= threshold]
        score = length(inliers)
        if score > best_score
            best_u = (M,c)
            best_score = score
            best_samples = inliers
        end
    end
    isnothing(best_samples) && throw(ErrorException("Ellipse not found: try an higher max_iters"))
    best_M,best_c = fit_orthogonal_ellipse(points_2d[best_samples])
    return best_score,best_M,best_c, best_samples
end

function affine(R::AbstractMatrix{T},t::AbstractVector{T}) where T
    M = MMatrix{4,4,T}(I)
    M[1:3,1:3] = R
    M[1:3,4] = t
    return SMatrix{4,4}(M)
end

function prepare_data(pcs,scan_coords; threshold=0.5)
    # Shuffle the data
    shuffle_idxs = randperm(length(pcs))
    shuffled_pcs = pcs[shuffle_idxs]
    shuffled_scan_coords = scan_coords[shuffle_idxs,:] # Shuffle the rows

    T_EB_data = SMatrix{4,4,Float64,16}[]
    x_data = []
    kept_indices = []
    k = 1
    for i::Int64 in ProgressBar(eachindex(shuffled_pcs))
        pc = shuffled_pcs[i]
        row = shuffled_scan_coords[i,:]
        # Rotations.jl uses intrinsic Euler, we use extrinsic XYZ
        R = SMatrix{3,3}(Rot.RotZYX(deg2rad.(row[6:-1:4])...))
        t = SVector{3}(row[1:3])
        T_EB = affine(R,t)

        X = map(x->SVector{2}(x[2:3]),pc)
        _,M,c,inliers = try
            orthogonal_ransac_ellipse(X;threshold=threshold)
        catch e
            @warn e
            continue
        end

        push!(T_EB_data, T_EB)
        push!(x_data, map(x->SVector{3,Float64}(x), pc[inliers]))
        push!(kept_indices, i)
        k+=1
    end
    shuffled_pcs = shuffled_pcs[kept_indices]
    return T_EB_data,x_data,shuffled_pcs
end

# Represent a cylinder with [r,v,c] where
# v is the direction: |v|=1
# c is the center: dot(c,v) = 0
# r is the radius: r > 0

# Reparametrize v as: v=(v+dv)/|v+dv|
# Reparametrize c as: c=(c+dc) - dot(c+dc,v)v

function normalize_v(v)
    v = SVector{3}(v)
    return v/norm(v)
end

function remove_v_component(x,v)
    x = SVector{3}(x)
    v = SVector{3}(v)
    return x - dot(x,v)*v
end

function project_point(T_EB, T_CE, x::AbstractVector{T}) where T
    x_h = SVector{4}(x[1], x[2], x[3], one(T))
    y_h = T_EB * T_CE * x_h
    return SVector{3}(@view(y_h[1:3]))
end

function get_tangent_basis(v)
    v = SVector{3}(v)
    F = eigen(Symmetric(I-v*transpose(v)))
    e1 = @view F.vectors[:,2]
    e2 = @view F.vectors[:,3]
    return SVector{3}(e1),SVector(e2)
end

function distance_from_cylinder(cylinder, x)
    v = @view cylinder[1:3]
    v = SVector{3}(v)
    c = @view cylinder[4:6]
    c = SVector{3}(c)
    r = cylinder[7]

    x = SVector{3}(x)
    u = x - c
    u_perp = u - dot(u, v) * v
    return norm(u_perp) - r
end

function _hat!(cache, v::AbstractVector{P}) where P
    # 1 4 7
    # 2 5 8
    # 3 6 9
    G1 = SMatrix{3,3}(0,0,0,0,0,1.0,0,-1.0,0)
    G2 = SMatrix{3,3}(0,0,-1.0,0,0,0,1.0,0,0)
    G3 = SMatrix{3,3}(0,1.0,0,-1.0,0,0,0,0,0)

    T = cache
    v = SVector{6}(v)
    T .= zero(P)
    θ = @view v[1:3]
    ρ = @view v[4:6]
    T[1:3,1:3] = θ[1]*G1 + θ[2]*G2 + θ[3]*G3
    T[1:3,4] = ρ
    return T
end

function Exp!(SE3, exp_cache, hat_cache, x)
    _hat!(hat_cache,x)
    exp!(SE3, exp_cache, hat_cache)
    return nothing
end

function residual(x,p)
    T_EB, point, cylinder, T_CE, exp_cache, hat_cache, SE3, e1, e2 = p

    v = @view cylinder[1:3]
    c = @view cylinder[4:6]
    r = cylinder[7]

    dRdt = @view x[1:6] #6
    dv   = @view x[7:8] #2
    dc   = @view x[9:10] #2

    
    Exp!(SE3, exp_cache, hat_cache, dRdt)
    T_CE = SMatrix{4,4}(T_CE)
    exp_cache = SMatrix{4,4}(exp_cache)
    new_T_CE = T_CE * exp_cache
    new_v = normalize_v(v + dv[1]*e1 + dv[2]*e2) # by construction satisfies dot(v,v)=1
    new_c = remove_v_component(c + dc[1]*e1 + dc[2]*e2, v) # by construction satisfies dot(c,v)=0
    new_cylinder = vcat(new_v, new_c, r)
    projected_point = project_point(T_EB,new_T_CE,point)
    return distance_from_cylinder(new_cylinder, projected_point)
end

function res_vec!(y,x,p)
    T_EB_data, x_data, cylinder, T_CE, hat_cache, exp_cache, SE3, e1, e2 = p
    
    hat_cache = get_tmp(hat_cache,x)
    exp_cache = get_tmp(exp_cache,x)
    y = get_tmp(y,x)
    
    k = 0
    for i in eachindex(T_EB_data)
        points = x_data[i]
        for j in eachindex(points)
            res_p = (T_EB_data[i], points[j], cylinder, T_CE, exp_cache, hat_cache, SE3, e1, e2)
            k+=1
            y[k] = residual(x,res_p)
        end
    end
    return y
end

function solve(data;max_iters=50)
    SE3 = SpecialEuclideanGroup(3)
    
    T_EB_data,x_data,cylinder,T_CE = data
    N = sum(x->size(x,1),x_data)
    hat_cache = DiffCache(zeros(4,4),10)
    exp_cache = DiffCache(zeros(4,4),10)
    
    
    p0 = zeros(10)
    e1, e2 = get_tangent_basis(cylinder[1:3])
    p = (T_EB_data, x_data, cylinder, T_CE, hat_cache, exp_cache, SE3, e1, e2)
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(res_vec!, resid_prototype = zeros(N)), p0, p,
        maxiters=1
    )
    solver = GaussNewton(
        autodiff=NonlinearSolve.AutoForwardDiff(), #chunksize=size(p0,1)
        linesearch=NonlinearSolve.LineSearch.BackTracking(autodiff=NonlinearSolve.AutoForwardDiff()),#chunksize=size(p0,1)
        linsolve=CholeskyFactorization()
    )
    sol = NonlinearSolve.solve(prob, solver)

    for i in 1:max_iters
        e1, e2 = get_tangent_basis(cylinder[1:3])
        p = (T_EB_data, x_data, cylinder, T_CE, hat_cache, exp_cache, SE3, e1, e2)
        prob = remake(prob; u0=p0,p=p)
        # 1 Newton step
        sol = NonlinearSolve.solve(prob, solver)
        # Update T_CE
        dRdt = @view sol.u[1:6]
        _exp_cache = get_tmp(exp_cache,sol.u)
        _hat_cache = get_tmp(hat_cache,sol.u)
        Exp!(SE3, _exp_cache, _hat_cache, dRdt)
        T_CE = T_CE*_exp_cache
        # Update cylinder
        dv   = @view sol.u[7:8]
        v = @view cylinder[1:3]
        v = normalize_v(v + dv[1]*e1 + dv[2]*e2) # by construction satisfies dot(v,v)=1
        dc   = @view sol.u[9:10]
        c = @view cylinder[4:6]
        c = remove_v_component(c + dc[1]*e1 + dc[2]*e2, v) # by construction satisfies dot(c,v)=0
        cylinder = vcat(v,c,cylinder[end])

        @info "Loss $(norm(sol.resid))"
        if norm(sol.u) < 1e-4
            break
        end
    end
    return T_CE, cylinder
end

export load_2d_scans, prepare_data, cost, residual, res_vec!


end

