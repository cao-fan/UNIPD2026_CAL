module CylinderCalibration

using CSV, DelimitedFiles, StatsBase, ProgressBars
using StaticArrays, LinearAlgebra, Rotations, LieGroups, Random, Manifolds
using Enzyme
using GLMakie

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

function orthogonal_ransac_ellipse(points_2d;threshold=1.0,max_iters=5000, max_radius=20, min_radius=10,min_samples=4)
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

function prepare_data(pcs,scan_coords; threshold=1.)
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
        R = SMatrix{3,3}(RotZYX(deg2rad.(row[6:-1:4])...))
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
    return v/norm(v)
end

function remove_v_component(x,v)
    return x - dot(x,v)*v
end

function project_point(T_EB, T_CE, x::AbstractVector{T}) where T
    x_h = SVector{4}(x[1], x[2], x[3], one(T))
    y_h = T_EB * (T_CE * x_h)
    return SVector{3}(y_h[1:3])
end

function get_tangent_basis(v)
    v = SVector{3}(v)
    F = eigen(I-v*transpose(v))
    e1 = F.vectors[:,2]
    e2 = F.vectors[:,3]
    return SVector{3}(e1),SVector(e2)
end

function distance_from_cylinder(cylinder, x)
    v = cylinder[1:3]
    c = cylinder[4:6]
    r = cylinder[7]

    u = x - c
    u_perp = u - dot(u, v) * v
    return norm(u_perp) - r
end

function residual(x,p)
    T_EB, point, cylinder, T_CE, e1, e2 = p

    v = cylinder[1:3]
    c = cylinder[4:6]
    r = cylinder[7]

    dRdt = x[1:6] #6
    dv = x[7:8] #2
    dc = x[9:10] #2

    SE3 = SpecialEuclideanGroup(3)
    se3 = LieAlgebra(SE3)
    perturb = exp(SE3, hat(se3, dRdt))
    new_T_CE = compose(SE3, T_CE, perturb)

    new_v = normalize_v(v + dv[1]*e1 + dv[2]*e2) # by construction satisfied dot(v,v)=1
    new_c = remove_v_component(c + dc[1]*e1 + dc[2]*e2, v) # by construction satisfied dot(c,v)=0
    new_cylinder = vcat(new_v,new_c,r)#vcat(new_v,new_c,new_r)

    projected_point = project_point(T_EB,new_T_CE,point)
    return distance_from_cylinder(new_cylinder, projected_point)
end

function cost_with_A_b(x, p)
    T_EB_data, x_data, cylinder, T_CE, e1, e2 = p
    dx = zero(x)
    S = 0.0
    JtJ = zeros(10,10)
    Jtb = zeros(10)
    N = size(T_EB_data,1)
    
    for i in eachindex(T_EB_data)
        T_EB = T_EB_data[i]
        measurements = x_data[i]
        M = size(measurements,1)
        for point in measurements
            res_p = (T_EB, point, cylinder, T_CE, e1, e2)
            dx .= 0.0 #zero gradient since it's accumulated
            resid = Enzyme.autodiff(Enzyme.ReverseWithPrimal, residual, Active, Duplicated(x,dx), Const(res_p))[2]
            S += resid^2 / N / M
            J = transpose(dx)
            JtJ += transpose(J)*J / N / M
            Jtb += transpose(J)*resid / N / M
        end
    end
    return S, JtJ, Jtb
end

using Infiltrator

function solve(data; max_iters=50)
    T_EB_data, x_data, initial_cylinder, initial_T_CE = data

    cylinder = initial_cylinder
    cylinder[1:3] = normalize_v(cylinder[1:3])
    cylinder[4:6] = remove_v_component(cylinder[4:6], cylinder[1:3])
    T_CE = initial_T_CE

    SE3 = SpecialEuclideanGroup(3)
    se3 = LieAlgebra(SE3)
    dx = zeros(10)
    old_loss = Inf
    lambda = 1e-3
    for i in 1:max_iters
        v = cylinder[1:3]
        c = cylinder[4:6]
        r = cylinder[7]

        e1, e2 = get_tangent_basis(v)
        # dx is the vector of perturbations in the tangent space
        p = (T_EB_data, x_data, cylinder, T_CE, e1, e2)
        obj_val, JtJ, Jtb = cost_with_A_b(dx,p)
        if obj_val < old_loss
            lambda /= 2
        else
            lambda *= 3
        end
        Δdx = (JtJ+lambda*I) \ -Jtb
        
        dSE3 = Δdx[1:6] #6
        dv   = Δdx[7:8] #2
        dc   = Δdx[9:10] #2
        @info "Iteration: $i" obj_val norm(Δdx)
        @debug "dSE3" dSE3[1:3] dSE3[4:6]
        @debug "dv,dc" dv dc
        # Update rigid transformation T_CE
        T_CE = compose(SE3, T_CE, exp(SE3, hat(se3, dSE3)))
        @debug "T_CE" vee(se3,log(SE3, T_CE))
        @debug "v,c,r" v c r
        # Update cylinder
        v = normalize_v(v + dv[1]*e1 + dv[2]*e2)
        c = remove_v_component(c + dc[1]*e1 + dc[2]*e2, v)
        cylinder = vcat(v,c,r)
        if norm(Δdx) < 1e-4
            break
        end
        old_loss = obj_val
    end
    return T_CE, cylinder
end

export load_2d_scans, prepare_data, cost_with_A_b, solve

end