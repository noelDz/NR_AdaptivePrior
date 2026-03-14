using Images
using Random
using BM3DDenoise
using CategoricalArrays, DataFrames, GLM, StableRNGs
using StatsBase
using LocalFilters

plot_font = "Computer Modern"
default(
    fontfamily=plot_font,
    linewidth=1, 
    framestyle=:box, 
    label=nothing, 
    grid=true
)

function image_plot(y,cmin,cmax,plttitle)
    if typeof(plttitle) == String
        #print("title detected")
        heatmap(y, color=:greys ,ticks=:false, showaxis = false,legend=false,aspect_ratio=1,clims=(cmin,cmax),title=plttitle)
    else
        heatmap(y, color=:greys ,ticks=:false, showaxis = false,legend=false,aspect_ratio=1,clims=(cmin,cmax))
    end
end

function getPositions(rangA, rangB)
    AP = Iterators.product(rangA, rangB) |> collect
    vcat(AP...)
end;

function genPlane(σ0::Float64, C1, C2, N::Int)
    Xrange = range(0.1, 1.0, length=N)
    Yrange = range(0.1, 1.0, length=N)
    n = length(Xrange)
    
    X = [(C1*x + C2*y) for x in Xrange, y in Yrange]
    X = X .- minimum(vcat(X...)) 
    
    if maximum(vcat(X...)) < 0
        X = X/minimum(vcat(X...))
    else 
        X = X/maximum(vcat(X...))
    end
    
    #if minimum(vcat(X...)) < 0
   # X = X - minimum(vcat(X...)) + 1
   # else 
        
   # end
    
    X .+= 1
    
    W = rand(Normal(0, σ0), n*n)
    X = reshape(X, (n, n))
    W = reshape(W, (n, n))
    Y = X + W
    
    return X, Y, W
end

function genVStripes(σ0::Float64,N::Int)
    X = []

    for i in range(1,N)
        if iseven(i)
            append!(X,ones(N))
        else
            append!(X,zeros(N))
        end
    end
    X .+= 1
    
    
    W = rand(Normal(0, σ0), N*N)
    X = reshape(X, (N, N))
    W = reshape(W, (N, N))
    Y = X + W
    
    return X, Y, W
end

function checkeredPattern_Blocks(σ,n)
    imband1 = []
    imband2 = []
    X = []
    #n = 100
    
    wSeg = ones(5,5)
    bSeg = zeros(5,5);
    
    
    for i in range(1,n)
        if iseven(i)
            push!(imband1,wSeg)
        else
            push!(imband1,bSeg)
        end
    end
    imband1 = mapreduce(permutedims, vcat, imband1)
    
    
    for i in range(1,n)
        if iseven(i)
            push!(imband2,bSeg)
        else
            push!(imband2,wSeg)
        end
    end
    imband2 = mapreduce(permutedims, vcat, imband2)
    
    
    for i in range(1,n)
        if iseven(i)
            push!(X,imband1)
        else
            push!(X,imband2)
        end
    end
    X = mapreduce(permutedims, vcat, X)
    X .+= 1
    
    N = size(X)[1]
    
    W = rand(Normal(0, σ), N*N)
    X = reshape(X, (N, N))
    W = reshape(W, (N, N))
    Y = X + W
    return X, Y, W
end

function blackwhite_bands(σ,n)
    X = []
    #n = 100
    
    wSeg = ones(n,5)
    bSeg = zeros(n,5)
       
    for i in range(1,Int(round(n/5)))
        if iseven(i)
            push!(X,wSeg)
        else
            push!(X,bSeg)
        end
    end
    X = mapreduce(permutedims, vcat, X)
    X .+= 1
    #print(size(X))
    #X = X[:n,:n]
    N = size(X)[1]
    
    W = rand(Normal(0, σ), N*N)
    X = reshape(X, (N, N))
    W = reshape(W, (N, N))
    Y = X + W
    return X,Y,W
end


function DiagBands(σ, f_dim)
    #f_dim = 64
    
    Xline = []
    wEl = ones(5);
    bEl = zeros(5);
    
    for i in range(1,round(f_dim))
        if iseven(i)
            push!(Xline,wEl)
        else
            push!(Xline,bEl)
        end
    end
    
    Xline = vcat(Xline...)
    lenline = length(Xline)
    
    X = []
    for i in range(1,f_dim)
            lin = vcat(Xline[i:lenline],Xline[1:i])
            push!(X,lin[1:f_dim])
    end
    
    X = mapreduce(permutedims, vcat, X)

    X = X.+1

    W = rand(Normal(0, σ), f_dim*f_dim)
    W = reshape(W, (f_dim, f_dim))
    #print(size(W))
    #print(size(X))
    Y = X + W
    return X, Y, W
end



# function r2_multilinReg(ImageMatrix)
#     nX,nY = size(ImageMatrix)   
#     nTot = nX+nY
#     pixValues = ImageMatrix

#     xV = collect(range(1,nX))
#     allind = Iterators.product(xV, xV) |> collect
#     allind = vcat(allind...)
#     triplets = [ [e[1],e[2],pixValues[e[1],e[2]]] for e in allind ]

#     xV = [e[1] for e in triplets];
#     yV = [e[2] for e in triplets];
#     zV = [e[3] for e in triplets];
#     data = DataFrame(xPos = float32.(xV), yPos = float32.(yV),pixelValue=float32.(zV))
    
#     regModel = lm(@formula(pixelValue ~ xPos+yPos), data)
    
    
#     mean(vcat(residuals(regModel)...).^2)      
# end

# function r2_multilinReg(ImageMatrix)
#     nX, nY = size(ImageMatrix)   
#     pixValues = ImageMatrix

#     xV = collect(range(1, nX))
#     allind = Iterators.product(xV, xV) |> collect
#     allind = vcat(allind...)
#     triplets = [[e[1], e[2], pixValues[e[1], e[2]]] for e in allind]

#     # Convert to Float64 instead of Float32
#     xV = [e[1] for e in triplets]
#     yV = [e[2] for e in triplets]
#     zV = [e[3] for e in triplets]
#     data = DataFrame(xPos = float.(xV),  # float() defaults to Float64
#                      yPos = float.(yV),
#                      pixelValue = float.(zV))
    
#     regModel = lm(@formula(pixelValue ~ xPos + yPos), data)
    
#     mean(residuals(regModel).^2)  # Simplified residual calculation
# end

function r2_multilinReg(ImageMatrix)
    # Convert input to Float64 if it isn't already
    ImageMatrix = Float64.(ImageMatrix)
    
    nX, nY = size(ImageMatrix)   
    pixValues = ImageMatrix

    xV = collect(range(1, nX))
    allind = Iterators.product(xV, xV) |> collect |> vec
    triplets = [[e[1], e[2], pixValues[e[1], e[2]]] for e in allind]

    # Create DataFrame with Float64
    data = DataFrame(
        xPos = getindex.(triplets, 1),
        yPos = getindex.(triplets, 2),
        pixelValue = getindex.(triplets, 3)
    )
    
    regModel = lm(@formula(pixelValue ~ xPos + yPos), data)
    mean(residuals(regModel).^2)
end


# function LinearPatch(y, L_patch,tau)
#     # Convert input to Float64 if needed
#     y = Float64.(y)
    
# #   L_y = size(y, 1)
#     nH, nV = size(y)

#     stepLen = 6
#     currMSE = Inf  # initialization
    
#     minEnt_patch = similar(y, L_patch, L_patch)
#     patch_index_i=0
#     patch_index_j=0
    
#     for i in Int.(floor.(range(1, L_y-L_patch, step=stepLen)))
#         for j in Int.(floor.(range(1, L_y-L_patch, step=stepLen)))
#             patch = y[i:i+L_patch-1, j:j+L_patch-1]
#             MSE = r2_multilinReg(patch)
#             most_frequent = mode(patch)
#             freq = countmap(patch)

#             #if MSE <= currMSE  && length(unique(patch)) > 64
#             if MSE <= currMSE  && length(unique(patch)) > tau
#                 patch_index_i = i
#                 patch_index_j = j
#                 currMSE = MSE
#                 minEnt_patch .= patch
#             end
#         end
#     end
#     #println("Final MSE: ", currMSE, "  NLE: ", sqrt(currMSE))
#     patch_index_i,patch_index_j, sqrt(currMSE)
# end

function LinearPatch(y, L_patch, tau)

    y = Float64.(y)
    nH, nV = size(y)

    stepLen = 6
    currMSE = Inf

    patch_index_i = 0
    patch_index_j = 0

    for i in 1:stepLen:(nH - L_patch + 1)
        for j in 1:stepLen:(nV - L_patch + 1)

            patch = y[i:i+L_patch-1, j:j+L_patch-1]

            MSE = r2_multilinReg(patch)

            if MSE <= currMSE && length(unique(patch)) > tau
                patch_index_i = i
                patch_index_j = j
                currMSE = MSE
            end
        end
    end

    return patch_index_i, patch_index_j, sqrt(currMSE)
end


# function NL_estimate_HV(Y)

#     nx,ny = size(Y)
    
#     polyOrder = 2
    
#     vPaths = VerticalPaths(Y)
#     pathSize = length(vPaths[1])
    
#     VlinCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in vPaths]
#     RegLinesV = vec([f.(collect(range(1,pathSize))) for f in VlinCoef])
#     residualsV = [RegLinesV[i] .- vPaths[i] for i in range(1,length(vPaths))]
#     ProbsV = [pvalue(LjungBoxTest(r, pathSize-1, polyOrder+1)) for r in residualsV]
    
#     hPaths = HorizontalPaths(Y)
#     HlinCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in hPaths]
#     RegLinesH = vec([f.(collect(range(1,pathSize))) for f in HlinCoef])
#     residualsH = [RegLinesH[i] .- hPaths[i] for i in range(1,length(hPaths))]
#     ProbsH = [pvalue(LjungBoxTest(r, pathSize-1, polyOrder+1)) for r in residualsH]

#     DPaths = ScanSlope1(Y)[1]
#     DPathInd = findall(v -> length(v) > 12, DPaths)
#     #print(length(DPathInd))
#     DPaths = DPaths[DPathInd]
#     DlinCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in DPaths]
#     RegLinesD = vec([DlinCoef[i].(collect(range(1,length(DPaths[i])))) for i in range(1,length(DPaths))])
#     residualsD = [RegLinesD[i] .- DPaths[i] for i in range(1,length(DPaths))]
#     ProbsD = [pvalue(LjungBoxTest(residualsD[i], length(DPaths[i])-1, polyOrder+1)) for i in range(1,length(residualsD))]

#     DM1Paths = ScanSlopeMinus1(Y)[1]
#     DM1PathInd = findall(v -> length(v) > 12, DM1Paths)
#     DM1Paths = DM1Paths[DM1PathInd]
#     DM1linCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in DM1Paths]
#     RegLinesDM1 = vec([DM1linCoef[i].(collect(range(1,length(DM1Paths[i])))) for i in range(1,length(DM1Paths))])
#     residualsDM1 = [RegLinesDM1[i] .- DM1Paths[i] for i in range(1,length(DM1Paths))]
#     ProbsDM1 = [pvalue(LjungBoxTest(residualsDM1[i], length(DM1Paths[i])-1, polyOrder+1)) for i in range(1,length(residualsDM1))]

#     th = 0.05
#     IndsV = findall(v -> v >th, ProbsV)
#     IndsH = findall(v -> v > th, ProbsH)
#     IndsD = findall(v -> v > th, ProbsD)
#     IndsDM1 = findall(v -> v > th, ProbsDM1)
#     numPasInds = sum(length.([IndsV,IndsH,IndsD,IndsDM1]))

#      while numPasInds  <= 5
#          th = th*.75
#       #   print("\n pCrit decreased to: \n", th)
#          IndsV = findall(v -> v > th, ProbsV)
#          IndsH = findall(v -> v > th, ProbsH)
#          # IndsD = findall(v -> v > th, ProbsD)
#          # IndsDM1 = findall(v -> v > th, ProbsDM1)
#          #numPasInds = sum(length.([IndsV,IndsH,IndsD,IndsDM1]))
#          numPasInds = sum(length.([IndsV,IndsH]))
#      end

#     print(numPasInds)

#     # resultV = [sigma_estimate_plus.(vPaths[IndsV])][1]
#     # resultH = [sigma_estimate_plus.(hPaths[IndsH])][1]
#     # resultD = [sigma_estimate_plus.(DPaths[IndsD])][1]
#     # resultDM1 = [sigma_estimate_plus.(DM1Paths[IndsDM1])][1]

#     resultV = [NLE_Auto(line,12) for line in vPaths[IndsV]]
#     resultH = [NLE_Auto(line,12) for line in hPaths[IndsH]]
#     #resultD = [NLE_Auto(line,12) for line in vPaths[IndsD]]
#     #resultDM1 = [NLE_Auto(line,12) for line in vPaths[IndsDM1]]
    
#     #PROBS = vcat(ProbsV[IndsV],ProbsH[IndsH],ProbsD[IndsD],ProbsDM1[IndsDM1])
#     #AllResults = vcat(vec(resultV),vec(resultH),vec(resultD),vec(resultDM1))
#     AllResults = vcat(vec(resultV),vec(resultH))
#     AllResults = vcat(AllResults...)
#     AllResults = filter(x -> !isnan(x), AllResults)
#     #mean(AllResults)
#     isempty(AllResults) && error("NL_estimate_HV: no valid NLE estimates")
#     return mean(AllResults)
# end

function NL_estimate_HV(Y)

    nx,ny = size(Y)
    
    polyOrder = 2
    
    vPaths = VerticalPaths(Y)
    pathSize = length(vPaths[1])
    
    VlinCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in vPaths]
    RegLinesV = vec([f.(collect(range(1,pathSize))) for f in VlinCoef])
    residualsV = [RegLinesV[i] .- vPaths[i] for i in range(1,length(vPaths))]
    ProbsV = [pvalue(LjungBoxTest(r, pathSize-1, polyOrder+1)) for r in residualsV]
    
    hPaths = HorizontalPaths(Y)
    HlinCoef = [curve_fit(Polynomial, Float32.(collect(range(1,length(y0)))), y0,polyOrder) for y0 in hPaths]
    RegLinesH = vec([f.(collect(range(1,pathSize))) for f in HlinCoef])
    residualsH = [RegLinesH[i] .- hPaths[i] for i in range(1,length(hPaths))]
    ProbsH = [pvalue(LjungBoxTest(r, pathSize-1, polyOrder+1)) for r in residualsH]

    th = 0.1
    IndsV = findall(v -> v >th, ProbsV)
    IndsH = findall(v -> v > th, ProbsH)
    numPasInds = sum(length.([IndsV,IndsH]))

     while numPasInds  <= 5
         th = th*.75
         IndsV = findall(v -> v > th, ProbsV)
         IndsH = findall(v -> v > th, ProbsH)
         numPasInds = sum(length.([IndsV,IndsH]))
     end

    resultV = [sigma_estimate(line) for line in vPaths[IndsV]]
    resultH = [sigma_estimate(line) for line in hPaths[IndsH]]
    
    AllResults = vcat(vec(resultV),vec(resultH))
    AllResults = vcat(AllResults...)
    AllResults = filter(x -> !isnan(x), AllResults)
    isempty(AllResults) && error("NL_estimate_HV: no valid NLE estimates")
    
    ##Method B: Estimates from the Residuals (requested modification)
    # We calculate the standard deviation of the residuals that passed the test
    resFilteredV = [std(residualsV[i]) for i in IndsV]
    resFilteredH = [std(residualsH[i]) for i in IndsH]
    allResid = filter(!isnan, vcat(resFilteredV, resFilteredH))

    return [mean(AllResults), mean(allResid)]
end


function PathSlopeMinus1(matrix, P0)
    nx,ny = size(matrix)
    path = [P0]
    all_positions = collect(Iterators.product(1:nx, 1:ny))
    pcurrent = P0
    
    for _ in 1:maximum([nx,ny])
        # First step (Left)
        if length(path) < length(all_positions)
            if pcurrent[1] == 1
                break
            else
                pcurrent = pcurrent .- (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .- (0, 1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScanSlopeMinus1(Y)
    nx, ny = size(Y)
    init_pos = [(nx, i) for i in 1:ny]
    init_pos2 = [(i, ny) for i in 1:nx]
    init_pos = vcat(init_pos, init_pos2)
    all_scan_positions = [PathSlopeMinus1(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagMinus1(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScanSlopeMinus1(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)
    
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], known_sigma,inf_criterion)
                                           
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################

    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
       # end
    end    
    return Xe2D, STDXe2D

end

function HorizontalPaths(matrix)
    nrows, ncols = size(matrix)
    path = [matrix[i, :] for i in 1:nrows]
    return path
end


function PSMestimHorizontal(Y, σ,inf_criterion);    
    yPaths = HorizontalPaths(Y)
    
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
     #stepSizes = Int.(ones(length(yPaths)))
   #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)
    

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)         
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)              
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end
    
    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    ########################################################################
    
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    
    Xe2D = copy(Y)
    STDXe2D = fill(σ^2, size(Y))
    
    pGauss = [pvalue(OneSampleTTest(xe[i] .- yPaths[i], 0)) for i in 1:length(yPaths)]
    
    for i in 1:length(yPaths)
        Xe2D[i, :] = xe[i]
        STDXe2D[i, :] = dxe[i]
    end
    
    return Xe2D, STDXe2D
end


# function VerticalPaths(matrix)
#     nrows, ncols = size(matrix)
#     path = [matrix[:, i] for i in 1:nrows]
#     return path
# end

function VerticalPaths(matrix)
    nrows, ncols = size(matrix)
    path = [matrix[:, j] for j in 1:ncols]
    return path
end
    

function PSMestimateVerticalPath(Y, σ,inf_criterion)
    yPaths = VerticalPaths(Y)
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)               
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ######################################################################## 
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    #xe = vcat(xe...) 
    #dxe = vcat(dxe...) 
    
    Xe2D = copy(Y)
    STDXe2D = fill(σ^2, size(Y))
    
    for i in 1:length(yPaths)
        Xe2D[:, i] = xe[i]
        STDXe2D[:, i] = dxe[i]
    end
    
    return Xe2D, STDXe2D
    #return xe, dxe
end

function PathSlope1(matrix, P0)
    nx,ny = size(matrix)
    path = [P0]
    all_positions = collect(Iterators.product(1:nx, 1:ny))
    pcurrent = P0
    
    for _ in 1:ny
        # First step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == nx
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .- (0, 1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScanSlope1(Y)
    nx,ny = size(Y)
    init_pos = [(1, i) for i in 1:ny]
    init_pos2 = [(i, ny) for i in 1:nx]
    init_pos = vcat(init_pos, reverse(init_pos2))
    all_scan_positions = [PathSlope1(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiag1(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScanSlope1(y)
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)
    #print(median(wSizes))

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)              
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]

    #goodPathPosi = findall(!iszero, length.(yPaths))
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathSlopeMinus1(matrix, P0)
    nx,ny = size(matrix)
    path = [P0]
    all_positions = collect(Iterators.product(1:nx, 1:ny))
    pcurrent = P0
    
    for _ in 1:maximum([nx,ny])
        # First step (Left)
        if length(path) < length(all_positions)
            if pcurrent[1] == 1
                break
            else
                pcurrent = pcurrent .- (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .- (0, 1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScanSlopeMinus1(Y)
    nx, ny = size(Y)
    init_pos = [(nx, i) for i in 1:ny]
    init_pos2 = [(i, ny) for i in 1:nx]
    init_pos = vcat(init_pos, init_pos2)
    all_scan_positions = [PathSlopeMinus1(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagMinus1(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScanSlopeMinus1(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)               
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################

    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathsSlope2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScansSlope2(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(1, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlope2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagS2(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlope2(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)                                 
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
      #  if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathsSlopeMinus2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # First step (Left)
        if length(path) < length(all_positions)
            if pcurrent[1] == 1
                break
            else
                pcurrent = pcurrent .+ (-1, 0)
            end
        end
        
        # Second Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScansSlopeMinus2(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(n, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlopeMinus2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagMinus2(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlopeMinus2(y)
    
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)
    #print(median(wSizes))
    
    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################    
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
   
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
       # end
    end    
    return Xe2D, STDXe2D

end

function PathsSlopeHalf(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                pcurrent = (pcurrent[1], n)
                push!(path, pcurrent)
                unique!(path)
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return path
end

function ScanSlopeHalf(Y)
    n = size(Y, 1)
    init_pos = [(1, i) for i in 1:n]
    init_pos2 = [(1, n) .+ (2k, 0) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(init_pos, init_pos2)
    
    all_scan_positions = [PathsSlopeHalf(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagSHalf(y, σ, inf_criterion)
    yPaths, positions,flatPositions  = ScanSlopeHalf(y)
    
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)
    #print(median(wSizes))
    
    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
       # end
    end    
    return Xe2D, STDXe2D

end

function PathsSlopeMinusHalf(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Downwards
        if length(path) < length(all_positions)
            if pcurrent[2] == n
                break
            else
                pcurrent = pcurrent .+ (0, 1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScansSlopeMinusHalf(Y)
    n = size(Y, 1)
    init_pos = [(1, i) for i in 1:n]
    init_pos2 = [(1, 1) .+ (2k, 0) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlopeMinusHalf(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagMinusHalf(y, σ,inf_criterion)
    yPaths, positions, flatPositions  = ScansSlopeMinusHalf(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathsSlope3(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
    end
    
    return unique(path)
end

function ScansSlope3(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(1, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlope3(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end


function PSMestimDiagS3(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlope3(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
        
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end


function PathsSlopeMinus3(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # First step (Left)
        if length(path) < length(all_positions)
            if pcurrent[1] == 1
                break
            else
                pcurrent = pcurrent .+ (-1, 0)
            end
        end
        
        # Second Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Third Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return unique(path)
end

function ScansSlopeMinus3(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(n, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlopeMinus3(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end


function PSMestimDiagSM3(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlopeMinus3(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end


function PathsSlope2_v2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # First Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        

    end
    
    return unique(path)
end

function ScansSlope2_v2(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(1, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlope2_v2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagS2_v2(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlope2_v2(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)

    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PSMout]
    #dxe = [output[2] for output in PSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
      #  if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end


function PathsSlopeMinus2_v2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First step (Left)
        if length(path) < length(all_positions)
            if pcurrent[1] == 1
                break
            else
                pcurrent = pcurrent .+ (-1, 0)
            end
        end
        
        # Second Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
            
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                break
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        

        end
    end
    
    return unique(path)
end


function ScansSlopeMinus2_v2(Y)
    n = size(Y, 1)
    init_pos = [(i, n) for i in 1:n]
    init_pos2 = [(n, n) .- (0, 2k) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlopeMinus2_v2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end

function PSMestimDiagMinus2_v2(y, σ,inf_criterion)
    yPaths, positions, flatPositions = ScansSlopeMinus2_v2(y)
    
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)
    #print(median(wSizes))
    #stepSizes = Int.(ones(length(yPaths)))
    
    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
        
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################    
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
   
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
      #  if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathsSlopeHalf_v2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Upwards
        if length(path) < length(all_positions)
            if pcurrent[2] == 1
                pcurrent = (pcurrent[1], n)
                push!(path, pcurrent)
                unique!(path)
            else
                pcurrent = pcurrent .+ (0, -1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)        
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
                push!(path, pcurrent)
                unique!(path)
            end
        end
    end
    
    return path
end

function ScanSlopeHalf_v2(Y)
    n = size(Y, 1)
    init_pos = [(1, i) for i in 1:n]
    init_pos2 = [(1, n) .+ (2k, 0) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(init_pos, init_pos2)
    
    all_scan_positions = [PathsSlopeHalf_v2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end


function PSMestimDiagSHalf_v2(y, σ, inf_criterion)
    yPaths, positions,flatPositions  = ScanSlopeHalf_v2(y)
    
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    stepSizes = ceil.(Int, wSizes ./ 6)
    #print(median(wSizes))
    #stepSizes = Int.(ones(length(yPaths)))
    
    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
             
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
      #  end
    end    
    return Xe2D, STDXe2D

end

function PathsSlopeMinusHalfV2(matrix, P0)
    n = size(matrix, 1)
    path = [P0]
    all_positions = collect(Iterators.product(1:n, 1:n))
    pcurrent = P0
    
    for _ in 1:n
        # First step (Right)
        
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
            end
        end
        
        # Step Downwards
        if length(path) < length(all_positions)
            if pcurrent[2] == n
                break
            else
                pcurrent = pcurrent .+ (0, 1)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        
        # Second step (Right)
        if length(path) < length(all_positions)
            if pcurrent[1] == n
                break
            else
                pcurrent = pcurrent .+ (1, 0)
                push!(path, pcurrent)
                unique!(path)
            end
        end
        

    end
    
    return unique(path)
end



function ScansSlopeMinusHalfV2(Y)
    n = size(Y, 1)
    init_pos = [(1, i) for i in 1:n]
    init_pos2 = [(1, 1) .+ (2k, 0) for k in 1:round(Int, n/2)]
    init_pos2 = init_pos2[1:end-1]
    init_pos = vcat(reverse(init_pos2), init_pos)
    
    all_scan_positions = [PathsSlopeMinusHalfV2(Y, ip) for ip in init_pos]
    flat_all_scan_positions = unique(vcat(all_scan_positions...))
    scan_values = [[Y[pos...] for pos in scan_pos] for scan_pos in all_scan_positions]
    
    return scan_values, all_scan_positions, flat_all_scan_positions
end


function PSMestimDiagMinusHalf_v2(y, σ,inf_criterion)
    yPaths, positions, flatPositions  = ScansSlopeMinusHalfV2(y)
    #wSizes = [getWindowSize(yP, σ) for yP in yPaths]
    wSizes = [Int(round(1.5*getWindowSize(yP,σ))) for yP in yPaths]
    #print(median(wSizes))
    stepSizes = ceil.(Int, wSizes ./ 6)    
   
    ########################################################################
    ### Parallel
    ParallelTable = distribute([[] for _ in procs()])
    
    @sync @distributed for i in 1:length(yPaths)
                        
                            xeFCF, σxeFCF = SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i], false,inf_criterion)
                            append!(localpart(ParallelTable)[1], [[i,xeFCF,σxeFCF]])                            
                        end

    ParallelTable = vcat(ParallelTable...);
    ParallelTable= sort(ParallelTable)
    xe = [r[2] for r in ParallelTable]  
    dxe = [r[3] for r in ParallelTable]  
    
    
    ########################################################################
    #PCSMout = [SWPSM(yPaths[i], σ, wSizes[i], stepSizes[i]) for i in 1:length(yPaths)]
    #xe = [output[1] for output in PCSMout]
    #dxe = [output[2] for output in PCSMout]
    
    goodPathPosi = findall(x -> x > 1, length.(yPaths))

    pGauss = zeros(length(yPaths))
    p = [pvalue(OneSampleTTest(xe[i] - yPaths[i])) for i in goodPathPosi]

    for i in 1:length(goodPathPosi)
        pGauss[i] = p[i]
        
    end
    #print("\n Size pGauss: ", size(pGauss))
    #print(maximum(goodPathPosi))
    
    Xe2D = copy(y)
    STDXe2D = fill(σ^2, size(y))
    
    #for i in 1:length(yPaths)
    for i in goodPathPosi
       # if pGauss[i] > pCritNorm
           pos = positions[i]
            xei = xe[i]
            stdi = dxe[i]
            
            for j in 1:length(pos)
                Xe2D[pos[j]...] = xei[j]
                STDXe2D[pos[j]...] = stdi[j]
            end
       # end
    end    
    return Xe2D, STDXe2D

end

function cPSM2D_Full(y, σ,inf_criterion,spikeTh)
    print("\n Computing Horizontal Scan... \n")
    hEstim,hEstimSTD = PSMestimHorizontal(y, σ,inf_criterion);    
                                        
    print("\n Computing Vertical Scan \n")
    vEstim,vEstimSTD = PSMestimateVerticalPath(y, σ,inf_criterion);    
    
    print("\n Computing Diagonal +1 Scan... \n")   
    D1E,D1Estd = PSMestimDiag1(y, σ,inf_criterion);    
    print("\n Computing Diagonal -1 Scan \n")   
    DM1E,DM1Estd  = PSMestimDiagMinus1(y, σ,inf_criterion);   

    print("\n Computing Diagonal +2 Scan... \n")   
    D2E,D2Estd  = PSMestimDiagS2(y, σ,inf_criterion); 

    print("\n Computing Diagonal -2 Scan (version 1)... \n")   
    DM2E,DM2Estd  = PSMestimDiagMinus2(y, σ,inf_criterion);

    print("\n Computing Diagonal +1/2 Scan (version 1)... \n")   
    DHE,DHEstd  = PSMestimDiagSHalf(y, σ,inf_criterion);

    print("\n Computing Diagonal -1/2 Scan (version 1)... \n")   
    DMHE,DMHEstd  = PSMestimDiagMinusHalf(y, σ,inf_criterion);  

    print("\n Computing Diagonal 3 Scan ... \n")   
    D3E,D3Estd = PSMestimDiagS3(y, σ,inf_criterion)
    
    print("\n Computing Diagonal -3 Scan... \n")   
    DM3E,DM3Estd= PSMestimDiagSM3(y, σ,inf_criterion)

    print("\n Computing Diagonal -2 (V2)Scan... \n")   
    DM2_v2_E,DM2_v2_Estd= PSMestimDiagMinus2_v2(y, σ,inf_criterion)

    print("\n Computing Diagonal 2 (V2)Scan... \n")   
    D2_v2_E,D2_v2_Estd= PSMestimDiagS2_v2(y, σ,inf_criterion)

    print("\n Computing Diagonal -1/2 (V2)Scan... \n")   
    DMH_v2_E,DMH_v2_Estd= PSMestimDiagMinusHalf_v2(y, σ,inf_criterion)

    print("\n Computing Diagonal 1/2 (V2)Scan... \n")   
    DH_v2_E,DH_v2_Estd= PSMestimDiagS2_v2(y, σ,inf_criterion)

    print("\n ------------------------ \n") 
    print("\n All Scans Computed! \n") 
    print("\n Initializing estimate aggregation step... \n") 


    ##########################################################
    #
    #         Inverese Variance Weighting Aggregation
    #
    ##########################################################
    
    scanEstims = [hEstim, vEstim,
             DM1E,D1E,
             DM2E,D2E,
             DMHE,DHE,
             D3E,DM3E,
             DM2_v2_E,D2_v2_E,
             DMH_v2_E,DH_v2_E
            ];
    
    scanVariances = [hEstimSTD,vEstimSTD,
                     DM1Estd,D1Estd,
                     DM2Estd,D2Estd,
                     DMHEstd,DHEstd,
                    D3Estd,DM3Estd,
                    DM2_v2_Estd,D2_v2_Estd,
                    DMH_v2_Estd,DH_v2_Estd
                ];
    
     @time begin
        
    Nx,Ny = size(y)
    nP = length(scanEstims)
        
    estimateTensor = scanEstims[1:nP]
    estimateVarTensor = scanVariances[1:nP]
    all_positions = collect(Iterators.product(1:Nx, 1:Ny));
       
    BLUEestims = map(all_positions) do pos
                i, j = pos       
                
                estim_fiber = [et[i, j] for et in estimateTensor]
                var_fiber = [vt[i, j] for vt in estimateVarTensor] 
    
                sort_ind_var = sortperm(var_fiber,rev = true) 
                var_fiber = var_fiber[sort_ind_var] #sorting from largest to smallest variance
                estim_fiber = estim_fiber[sort_ind_var] #sorting from largest to smallest variance
                n_estims = length(scanEstims)
            
                spike_positions = findall(x -> x >= σ^2, var_fiber)    
    
                if length(spike_positions) >= spikeTh #if there is at least one spike...
                     estim_fiber=[y[i,j]] #only keep the spike
                     var_fiber=[σ^2]   #only keep the spike
                else
                     estim_fiber=deleteat!(estim_fiber, spike_positions) #only keep the spike
                     var_fiber=deleteat!(var_fiber, spike_positions)   #only keep the spike
                end
                
                return BLUEindep(estim_fiber, sqrt.(var_fiber)), BLUEvariance(sqrt.(var_fiber))
    end
    
    
    XEBLUE= reshape([e[1] for e in BLUEestims], (Nx, Ny));
    BLUEvar = reshape([e[2] for e in BLUEestims], (Nx, Ny));
    
    end
    print("\n Aggregation Step: Done! \n")
    print("\n cPSM2D Estimate computation: Done\n")
    return(scanEstims,scanVariances,XEBLUE,BLUEvar)
end

function cPSM2D(y, σ,inf_criterion,spikeTh)
    print("\n Computing Horizontal Scan... \n")
    hEstim,hEstimSTD = PSMestimHorizontal(y, σ,inf_criterion);    
    
    print("\n Computing Vertical Scan \n")
    vEstim,vEstimSTD = PSMestimateVerticalPath(y, σ,inf_criterion);    
    
    print("\n Computing Diagonal +1 Scan... \n")   
    D1E,D1Estd = PSMestimDiag1(y, σ,inf_criterion);    
   
    print("\n Computing Diagonal -1 Scan \n")   
    DM1E,DM1Estd  = PSMestimDiagMinus1(y, σ, inf_criterion);   

    print("\n ------------------------ \n") 
    print("\n All Scans Computed! \n") 
    print("\n Initializing estimate aggregation step... \n") 


    ##########################################################
    #
    #         Inverse Variance Weighting Aggregation
    #
    ##########################################################
    
    scanEstims = [
            hEstim, vEstim,
             DM1E,D1E
            ];
    
    scanVariances = [hEstimSTD,vEstimSTD,
                     DM1Estd,D1Estd
                ];
    
    @time begin
        
    Nx,Ny = size(y)
    nP = length(scanEstims)
        
    estimateTensor = scanEstims[1:nP]
    estimateVarTensor = scanVariances[1:nP]
    all_positions = collect(Iterators.product(1:Nx, 1:Ny));
       
    BLUEestims = map(all_positions) do pos
                i, j = pos       
                
                estim_fiber = [et[i, j] for et in estimateTensor]
                var_fiber = [vt[i, j] for vt in estimateVarTensor] 
    
                sort_ind_var = sortperm(var_fiber,rev = true) 
                var_fiber = var_fiber[sort_ind_var] #sorting from largest to smallest variance
                estim_fiber = estim_fiber[sort_ind_var] #sorting from largest to smallest variance
                n_estims = 4
            
                spike_positions = findall(x -> x >= σ^2, var_fiber)    
    
                if length(spike_positions) >= spikeTh #if there is at least one spike...
                     estim_fiber=[y[i,j]] #only keep the spike
                     var_fiber=[σ^2]   #only keep the spike
                else
                     estim_fiber=deleteat!(estim_fiber, spike_positions) #only keep the spike
                     var_fiber=deleteat!(var_fiber, spike_positions)   #only keep the spike
                end
                
                normalized_precision  = inv.(var_fiber)/ sum(inv.(var_fiber))
                
                # if length(var_fiber) > 1
                #         n_estims = 1
                #         while sum(normalized_precision[1:n_estims]) < 0.99
                #             n_estims = n_estims + 1
                #         end
                #     else
                #         n_estims = length(var_fiber)
                # end
                #n_estims = length(var_fiber)
                return BLUEindep(estim_fiber, sqrt.(var_fiber)), BLUEvariance(sqrt.(var_fiber))
    end
    
    
    XEBLUE= reshape([e[1] for e in BLUEestims], (Nx, Ny));
    BLUEvar = reshape([e[2] for e in BLUEestims], (Nx, Ny));
    
    end
    print("\n Aggregation Step: Done! \n")
    print("\n cPSM2D Estimate computation: Done\n")
    return(scanEstims,scanVariances,XEBLUE,BLUEvar)
end

function aggregateScans(y,scanEstims,scanVariances,spikeThr,NLE)
    Nx,Ny = size(y)
    nP = length(scanEstims)
        
    estimateTensor = scanEstims[1:nP]
    estimateVarTensor = scanVariances[1:nP]
    all_positions = collect(Iterators.product(1:Nx, 1:Ny));
       
    BLUEestims = map(all_positions) do pos
                i, j = pos       
                
                estim_fiber = [et[i, j] for et in estimateTensor]
                var_fiber = [vt[i, j] for vt in estimateVarTensor] 
    
                sort_ind_var = sortperm(var_fiber,rev = true) 
                var_fiber = var_fiber[sort_ind_var] #sorting from largest to smallest variance
                estim_fiber = estim_fiber[sort_ind_var] #sorting from largest to smallest variance              
            
                spike_positions = findall(x -> x >= NLE^2, var_fiber)    
    
                if length(spike_positions) >= spikeThr #if there are at least spikeThr spike...
                     estim_fiber=[y[i,j]] #leave point untouched
                     var_fiber=[NLE^2]   #assign this variance
                else
                     estim_fiber=deleteat!(estim_fiber, spike_positions) #remove spikes from the fiber
                     var_fiber=deleteat!(var_fiber, spike_positions)   
                end

                normalized_precision  = sort(inv.(var_fiber)/ sum(inv.(var_fiber)),rev=true)

                #n_estims = minimum([length(estim_fiber),2])
                n_estims=1
                if length(var_fiber) > 1
                      while sum(normalized_precision[1:n_estims]) < .97
                        n_estims = n_estims + 1
                    end
                end

                return BLUEindep(estim_fiber[1:n_estims], sqrt.(var_fiber[1:n_estims])), 
                BLUEvariance(sqrt.(var_fiber[1:n_estims])),
                n_estims
                # end
    end
    
    XEBLUE= reshape([e[1] for e in BLUEestims], (Nx, Ny));
    BLUEvar = reshape([e[2] for e in BLUEestims], (Nx, Ny));
    Nest = reshape([e[3] for e in BLUEestims], (Nx, Ny));

    N_th = maximum(vcat(Nest))
    hiErr_pos = findall(x-> N_th > x > 1, Nest)
    
    return(XEBLUE,BLUEvar,Nest)
end

function getNoisyimage(path,addedNoiselevel,NLE_patchSize,seed,estimate_bool,FullImage)
    
    print(split(path,'.')[end])
    obj=[]
    
    if typeof(seed)==Int
        Random.seed!(seed)
    else
        seed = "None"
    end
    if path[end-3:end]==".csv"
        print("CSV file... \n")
        obj = readdlm(path,',')
    end
    if split(path,'.')[end]=="tsv"
       print("TSV found")
       obj = reverse(readdlm(path),dims=1)
    end
    if split(path,'.')[end]=="tiff" 
       obj = load(path)
       obj = reverse(load(path), dims=1)
    end
    if split(path,'.')[end]=="tif" 
       obj = load(path)
       obj = reverse(load(path), dims=1)
    end
    
    # Convert RGB to grayscale if needed
    if eltype(obj) <: Colorant
        obj = Gray.(obj)
    end
    
    if length(size(obj)) == 3
        n1,n2,n3 = size(obj)
        obj = obj[1,1:n2,1:n3]
    else
        n1,n2 = size(obj)
        obj = obj[1:n1,1:n2];
    end
    
    x = Float64.(obj)
    x = x/maximum(x)
    
    nH,nV=size(x)
    
    print("\n -----------------------------------")
    print("\n Experiment parameters: ")
    print("\n Image Size = ", size(x))
    print("\n Image Min,Max = ", [minimum(x),maximum(x)])
    


    if addedNoiselevel > 0
        w = rand(Normal(0, addedNoiselevel), nH, nV);        
        y = x + w
        
        psnr0 = assess_psnr(y,x)
        
        snri0 =  getSNRI(vcat(x...),vcat(y...),vcat(y...))
        print("\n Added Noise: ",addedNoiselevel,"\n")
        if FullImage==false
            print("\nProcessing Image Segment")
            #Trimming (ONLY FOR QUICK TESTS)
            i0=200
            hSize =100
            x = x[i0:i0+hSize,i0:i0+hSize]
            y = y[i0:i0+hSize,i0:i0+hSize]
            w = w[i0:i0+hSize,i0:i0+hSize]
        end

        print("Size (y): ",size(x))
    else
        y=[]
        print("\n Added Noise: ",0.0,"\n")
        i0=200
        hSize =200
        x = x[i0:i0+hSize,i0:i0+hSize]
        y = x
        w= zeros(nH,nV)
    end
    
    
    patch_index_i,patch_index_j, σeMSE = LinearPatch(y,NLE_patchSize,0);
    print("\n", [patch_index_i,patch_index_j])
    print("\n", [patch_index_i+NLE_patchSize-1,patch_index_j+NLE_patchSize-1])
    LP = y[patch_index_i:patch_index_i+NLE_patchSize-1, patch_index_j:patch_index_j+NLE_patchSize-1]
    
    if estimate_bool == true
        σeANT,σeCLAS = NL_estimate_HV(LP);
    else 
        σeANT = addedNoiselevel
        σeCLAS =σeANT
    end

    return w,x, y, σeANT,σeCLAS, patch_index_i, patch_index_j
end



function outlierPixels(y_SPSM,nEstimators_SPSM,th)
    np_X,np_Y = size(y_SPSM);
    npix =  np_X * np_Y
   # N_TH = median(vcat(nEstimators_SPSM...))
   # ERRposi =  findall(x -> x <= th,nEstimators_SPSM);

    TextutedPix_Ind = findall(x->x<=th,nEstimators_SPSM);
            
    print("Percentage of high-texture Pixels: ",
     round(length(TextutedPix_Ind)*100/npix,digits=3) ,"%")
            
    dummy_mat = zeros(np_X,np_Y)
    for elem in TextutedPix_Ind
        i,j = elem[1],elem[2]
        dummy_mat[i,j] = 1
    end;

    return(TextutedPix_Ind,image_plot(dummy_mat,0,1,""))

end



function outlierPixels_Var(y_SPSM,VAR_SPSM,th)
    np_X,np_Y = size(y_SPSM);
    npix =  np_X * np_Y
    # q1, q3 = quantile(vcat(VAR_SPSM...), [0.25, 0.75])
    # iqr = q3 - q1
    # #th = q3 + 1. * iqr
    # th = quantile(vcat(VAR_SPSM...),.25)
    #th = maximum(Xe_SmoothRegion[2])
    print("\n_____________________")
    print("\n high var threshold: ",th)
    
    TextutedPix_Ind = findall(x->x>th,VAR_SPSM);
    print("\nPercentage of High Variance Pixels: ",100*length(TextutedPix_Ind)/npix)
            
    dummy_mat = zeros(np_X,np_Y)
    for elem in TextutedPix_Ind
        i,j = elem[1],elem[2]
        dummy_mat[i,j] = 1
    end;

    return(TextutedPix_Ind,image_plot(dummy_mat,0,1,""))

end

function All_texturedPatches(yPSM,TextutedPix_Ind,patchSize)
    np_X,np_Y = size(yPSM)
    half_Size = Int(floor(patchSize/2))
    AllPatches = []
    for elem in TextutedPix_Ind
        i,j = elem[1],elem[2]
         if i-half_Size > 1 && i+half_Size < np_X && j-half_Size > 1 && j+half_Size < np_Y
            patch = yPSM[i-half_Size:i+half_Size,j-half_Size:j+half_Size] 
            AllPatches = push!(AllPatches,patch)
        end
    end;
    return AllPatches
end


function search_Region(pos, half_size::Int, nrows::Int, ncols::Int)
    i, j = pos[1],pos[2]
    rows = collect(max(1, i-half_size):min(nrows, i+half_size))
    cols = collect(max(1, j-half_size):min(ncols, j+half_size))

    inds = Iterators.product(rows,cols) |> collect
    vcat(inds...)
end


# ================================================================
# Draw square overlay (robust for heatmaps)
# ================================================================
function draw_square2!(
    center::Tuple{Int,Int},
    half_size::Int;
    color = :red,
    lw::Real = 3,
    alpha::Real = 0.99)
    # center = (row, col)
    i, j = center
    x0, y0 = j, i

    ε = 0.5  # critical pixel-center offset

    x = [
        x0 - half_size - ε,
        x0 + half_size + ε,
        x0 + half_size + ε,
        x0 - half_size - ε
    ]
    y = [
        y0 - half_size - ε,
        y0 - half_size - ε,
        y0 + half_size + ε,
        y0 + half_size + ε
    ]

    plot!(
        Shape(x, y);
        fill = false,
        linecolor = color,
        linewidth = lw,
        alpha = alpha
    )

    return nothing
end

# ================================================================
# K / ε-nearest patch extraction (FIXED & CONSISTENT)
# ================================================================
function get_K_NearestPatches(
    xPSM::AbstractMatrix,
    X::AbstractMatrix,
    Y::AbstractMatrix,
    ref_Ind::Int,
    TextutedPix_Ind,
    patchSize::Int,
    sigma_e;
    q::Float64 = 9.0,
    search_radius::Int = 30,
    )


    TextutedPix_Ind_Tup = [(e[1], e[2]) for e in TextutedPix_Ind]
    hSize = (patchSize-1) ÷ 2
    np_X, np_Y = size(xPSM)

    ref_p = TextutedPix_Ind_Tup[ref_Ind]

    # bounds check
    if ref_p[1] - hSize < 1 || ref_p[1] + hSize > np_X ||
       ref_p[2] - hSize < 1 || ref_p[2] + hSize > np_Y
        error("Reference patch out of bounds")
    end

    # reference patches
    ref_patch = @views xPSM[
        ref_p[1]-hSize:ref_p[1]+hSize,
        ref_p[2]-hSize:ref_p[2]+hSize
    ]
    ref_patch_GT = @views X[
        ref_p[1]-hSize:ref_p[1]+hSize,
        ref_p[2]-hSize:ref_p[2]+hSize
    ]
    ref_patch_Y = @views Y[
        ref_p[1]-hSize:ref_p[1]+hSize,
        ref_p[2]-hSize:ref_p[2]+hSize
    ]

    # local_var = var(vcat(ref_patch...))
    # multiplier = 1.0 + (sigma_e^2 / (local_var + 1e-6))
    # q = q*multiplier
    # print("\nq:",q)
    # neighbourhood restriction
    surrounding = vcat(search_Region(ref_p, search_radius, np_X, np_Y)...)
    valid_neighbours = intersect(TextutedPix_Ind_Tup, surrounding)

    Sor_Patches = Matrix{eltype(xPSM)}[]
    VAL_IND = Tuple{Int,Int}[]

    for (i, j) in valid_neighbours
        if i - hSize >= 1 && i + hSize <= np_X &&
           j - hSize >= 1 && j + hSize <= np_Y
            patch = @views Y[i-hSize:i+hSize, j-hSize:j+hSize]   ###COMPUTING DISTANCES IN THE ORIGINAL IMAGE 
            push!(Sor_Patches, patch)
            push!(VAL_IND, (i, j))
        end
    end
    push!(Sor_Patches,ref_patch)
    push!(VAL_IND,(ref_p[1],ref_p[2]))

    # distances
    nP = length(Sor_Patches)
    D = Vector{Float64}(undef, nP)
    for k in 1:nP
        @inbounds D[k] = sqrt(sum(abs2, ref_patch_Y .- Sor_Patches[k]))
    end

    # ε-nearest selection + SORT (CRITICAL FIX)
    #thDistance = quantile(D, q)
    thDistance = q * (sigma_e^2) * (patchSize^2) 
    keep = findall(<(thDistance), D)

    Dk = D[keep]
    p = sortperm(Dk)

    closest_patches = Sor_Patches[keep][p]
    closest_patches_ind = VAL_IND[keep][p]

    return ref_p,
           ref_patch,
           ref_patch_GT,
           ref_patch_Y,
           Dk[p],
           closest_patches,#[1:minimum([patchSize^2,length(closest_patches)])],
           closest_patches_ind
end


# function to draw square over heatmap
function draw_square2!(center::Tuple{Int,Int}, half_size::Int, color=:red, lw=3)
    i, j = center  # i=row, j=column
    # heatmap uses (x=j, y=i) with row 1 at bottom
    x_center, y_center = j, i

    x = [x_center - half_size, x_center + half_size, x_center + half_size,
         x_center - half_size, x_center - half_size]
    y = [y_center - half_size, y_center - half_size, y_center + half_size,
         y_center + half_size, y_center - half_size]

    plot!(x, y, color=color, lw=lw,alpha=.99)
end


# ============================================================
# PCA on a set of neighboring patches (columns = patches)
# ============================================================

"""
    compute_eigenPatch(neighbors::Vector{Matrix{Float64}})

Compute PCA on a set of neighboring patches.
- Columns of X correspond to vectorized neighbor patches.
Returns (mean_patch, eigenpatches_matrix, eigenpatches_list, eigenvalues)
"""

function compute_eigenPatch(neighbors::Vector{Matrix{Float64}})
    h, w = size(neighbors[1])
    d = h * w
    M = length(neighbors)

    # Stack vectorized patches as columns (d × M)
    X = hcat(vec.(neighbors)...)

    # Mean patch (d × 1)
    μ = mean(X, dims=2)
    
    # Centered data
    Xc = X .- μ

    # SVD
    U, S, Vt = svd(Xc; full=false)

    # Effective PCA rank
    r = size(U, 2)

    # Eigenvalues
    eigenvalues = (S .^ 2) ./ (M - 1)

    # Eigenpatches
    eigenpatches = U                    # (d × r)

    # Visualization list (ONLY up to r)
    eigenpatches_list = [
        reshape(eigenpatches[:, i], h, w) for i in 1:r
    ]
    @assert size(eigenpatches, 2) ≤ d
    return reshape(μ, h, w), eigenpatches, eigenpatches_list, eigenvalues
end


# ============================================================
# Reconstruction using the first k eigenpatches
# ============================================================

"""
    reconstruct(patch, mean_patch, eigenpatches, k)

Reconstruct a patch using the first k eigenpatches.
"""
function reconstruct(
    patch::AbstractMatrix{<:Real},
    mean_patch::AbstractMatrix{<:Real},
    eigenpatches::AbstractMatrix{<:Real},
    k::Int)
    h, w = size(patch)
    x = vec(patch)
    μ = vec(mean_patch)

    U_k = eigenpatches[:, 1:k]

    # Project onto PCA subspace
    coeffs = U_k' * (x - μ)

    # Reconstruct
    x_recon = μ + U_k * coeffs

    return reshape(x_recon, h, w)
end


# # ============================================================
# # Minimum AIC reconstruction
# # ============================================================

# """
#     min_AIC_PatchReconstruction(refPatch, NLE, closest_patches)

# Compute reconstructions and return the one corresponding to minimum AIC.
# - Returns:
#     reconstructions: all reconstructions
#     aic_rec: AIC values
#     best_recon: reconstruction with minimal AIC
# """
# function min_AIC_PatchReconstruction(refPatch::AbstractMatrix{<:Real},
#                                      NLE::Float64,
#                                      closest_patches::Vector{Matrix{Float64}})

#     # --- PCA ---
#     mean_patch, V_k, _, eigenvalues = compute_eigenPatch(closest_patches)
#     D, M = size(V_k)
#     ncomp = M
#     N = length(closest_patches)

#     # --- Reconstruct 1..ncomp components ---
#     reconstructions = [reconstruct(refPatch, mean_patch, V_k, comp) for comp in 1:ncomp]

#     # --- AIC calculation ---
#     #nparams = D + D*i + N*i
#     nparams = D + D*i - (i^2)/2 + i/2

#     aic_rec = [N*(sum(eigenvalues[i+1:ncomp]))/(NLE^2) + 2*(nparams) + (N*D*log(2π*NLE^2))
#                for i in 1:ncomp-1]

#     # --- Pick best reconstruction ---
#     best_recon = reconstructions[argmin(aic_rec)]

#     return reconstructions, aic_rec, best_recon
# end


# function AIC_weighted_reconstruction2(refPatch::AbstractMatrix{<:Real},
#                                      NLE::Float64,
#                                      closest_patches::Vector{Matrix{Float64}})

#     # --- PCA decomposition ---
#     mean_patch, V_k, _, eigenvalues = compute_eigenPatch(closest_patches)

#     D, r = size(V_k)          # D = patch dimension, r = PCA rank
#     ncomp = r
#     N = length(closest_patches)

#     # --- Reconstruct using 1..ncomp components ---
#     reconstructions = [
#         reconstruct(refPatch, mean_patch, V_k, comp) for comp in 1:ncomp
#     ]

#     # --- Compute AIC for each reconstruction ---
    

#     aic_rec = [
#         N * sum(eigenvalues[i+1:ncomp]) / (NLE^2) +
#         2 * (D + (D*i - (i^2)/2 - i/2) + N*i) +
#         (N * D * log(2π * NLE^2))
#         for i in 1:ncomp-1
#     ]

#     # --- AIC minimum ---
#     AIC_min = minimum(aic_rec)
#     C = argmin(aic_rec)

#     # --- Select models with ΔAIC < 1 ---
#     ΔAIC = aic_rec .- AIC_min
#     idx_range = findall(ΔAIC .< 1000)

#     # --- Compute weights only on selected models ---
#     weights = exp.(-0.5 .* ΔAIC[idx_range])
#     weights ./= sum(weights)

#     # --- Weighted reconstruction ---
#     h, w = size(refPatch)
#     weighted_recon = zeros(Float64, h, w)
#     for (j, i) in enumerate(idx_range)
#         weighted_recon .+= weights[j] .* reconstructions[i]
#     end

#     return reconstructions, aic_rec, weights, weighted_recon, C, idx_range
# end


function AIC_weighted_reconstruction2(refPatch::AbstractMatrix{<:Real},
    NLE::Float64,
    closest_patches::Vector{Matrix{Float64}})

# --- PCA decomposition ---
mean_patch, V_k, _, eigenvalues = compute_eigenPatch(closest_patches)

D, r = size(V_k)          # D = patch dimension, r = max PCA rank
ncomp = r
N = length(closest_patches)
σ2 = NLE^2

# --- Reconstruct using 1..ncomp components ---
reconstructions = [
reconstruct(refPatch, mean_patch, V_k, comp) for comp in 1:ncomp
]
aic_rec = Float64[]
for i in 1:ncomp
    # 1. Data Fidelity (Residual Sum of Squares / σ²)
    # eigenvalues[j] = \hat{λ}²_j / (N - 1) -> we recover the raw sum
    rss_term = ((N - 1) * sum(eigenvalues[i+1:ncomp])) / σ2
    
    # 2. Complexity Penalty: 2 * (total degrees of freedom)
    # p(i) = D (for the mean vector) + [D*i - i*(i+1)/2] (for the orthonormal basis)
    # We EXCLUDE (N*i) because Z is profiled out in the surrogate likelihood.
    
    df_mean = D
    df_subspace = (D * i) - (i * (i + 1) / 2)
    df_sing_vectors = i
    
    # 3. Log-likelihood Constant
    # N varies; it scales the likelihood to the sample size.
    const_term = N * D/2 * log(2π * σ2)
    NParams = df_mean + df_subspace + df_sing_vectors+ 1+ N*i

    penalty_term = 2 * NParams
    penalty_term_BIC = log(D)*(NParams)
    #Correction_Term = (2*NParams*(NParams+1))/(N-NParams-1)
    
    push!(aic_rec,const_term+ rss_term + penalty_term)
end

# aic_rec = [
# N * sum(eigenvalues[i+1:ncomp]) / σ2 +
# #2 * (D + D*i - (i^2)/2 - (i/2) + N ) +          # NEW DERIVATION 
# 2 * (D + (D*i - (i^2)/2 - i/2) + N+(N*i)) +     # PREVIOUS
# (N * D * log(2π * σ2))
# for i in 1:ncomp]

# --- AIC minimum ---
AIC_min = minimum(aic_rec)
C = argmin(aic_rec)

# --- Select models with ΔAIC < 1000 ---
ΔAIC = aic_rec .- AIC_min
idx_range = findall(ΔAIC .< 2)

# --- Compute weights only on selected models ---
weights = exp.(-0.5 .* ΔAIC[idx_range])
weights ./= sum(weights)

# --- Weighted reconstruction and Variance calculation ---
h, w = size(refPatch)
weighted_recon = zeros(Float64, h, w)
weighted_var = zeros(Float64, h, w)

# First pass: Compute the weighted mean (weighted_recon)
for (j, k) in enumerate(idx_range)
weighted_recon .+= weights[j] .* reconstructions[k]
end

# Second pass: Compute the Burnham-Anderson Variance
for (j, k) in enumerate(idx_range)
# 1. Within-model variance: diag(σ² * Uk * Uk')
# We calculate the squared norm of rows for the first k eigenvectors
Vk_subset = V_k[:, 1:k]
within_var_vec = σ2 .* sum(Vk_subset.^2, dims=2)
within_var_patch = reshape(within_var_vec, h, w)

# 2. Across-model variance: (x_k - x_avg)²
across_var_patch = (reconstructions[k] .- weighted_recon).^2

# 3. Aggregate
weighted_var .+= weights[j] .* (within_var_patch .+ across_var_patch)
end

# Returns original objects + the new weighted_var
return reconstructions, aic_rec, weights, weighted_recon, C, idx_range, weighted_var
end 


# function AIC_weighted_reconstruction_PPCA(refPatch::AbstractMatrix{<:Real},
#     NLE::Float64,
#     closest_patches::Vector{Matrix{Float64}})

# # --- PCA decomposition ---
# mean_patch, V_k, _, eigenvalues = compute_eigenPatch(closest_patches)

# D, r = size(V_k)              # D = patch dimension
# ncomp = r
# N = length(closest_patches)
# σ2 = NLE^2

# # --- Reconstruct using 1..ncomp components ---
# reconstructions = [
# reconstruct(refPatch, mean_patch, V_k, comp) for comp in 1:ncomp
# ]

# # --- Precompute cumulative eigenvalue sums ---
# # (for efficient likelihood computation)
# total_eigs = sum(eigenvalues)
# cumsum_eigs = cumsum(eigenvalues)

# # --- Compute AIC for k = 1:(ncomp-1) ---
# aic_rec = Float64[]

# for k in 1:(ncomp-1)

# # ---- Log-determinant term ----
# logdet_term = sum(log.(eigenvalues[1:k])) +
# (D - k) * log(σ2)

# # ---- Trace term ----
# residual_sum = total_eigs - cumsum_eigs[k]

# trace_term = k + residual_sum / σ2

# # ---- Deviance (-2 log-likelihood up to constant) ----
# deviance = N * (logdet_term + trace_term)

# # ---- Parameter count (σ known) ----
# mk = D*k - k*(k-1)/2

# # ---- AIC ----
# AIC_k = deviance + 2*mk

# push!(aic_rec, AIC_k)
# end

# # --- AIC minimum ---
# AIC_min = minimum(aic_rec)
# C = argmin(aic_rec)

# # --- Select models with ΔAIC < 1 ---
# ΔAIC = aic_rec .- AIC_min
# idx_range = findall(ΔAIC .< 1)

# # --- Compute weights only on selected models ---
# weights = exp.(-0.5 .* ΔAIC[idx_range])
# weights ./= sum(weights)

# # --- Weighted reconstruction ---
# h, w = size(refPatch)
# weighted_recon = zeros(Float64, h, w)
# for (j, i) in enumerate(idx_range)
# weighted_recon .+= weights[j] .* reconstructions[i]
# end

# return reconstructions, aic_rec, weights, weighted_recon, C, idx_range
# end


"""
    valid_centers(centers, L, nY, nX)

Return the positions (indices in the input list) of centers 
whose L×L squares are fully contained in the rectangle 
(1,1) -- (1,nX) -- (nY,1) -- (nY,nX).

- `centers` can be a vector of `Tuple`s or `CartesianIndex`.
- `nY`, `nX` define the rectangle size.
"""
function valid_centers(centers, L::Int, nY::Int, nX::Int)
    half = div(L, 2)
    offset = (L % 2 == 0) ? (half - 1) : half

    valid_idx = Int[]
    for (k, c) in enumerate(centers)
        i, j = Tuple(c)
        if i - offset > 1 && j - offset > 1 &&
           i + half ≤ nY && j + half ≤ nX
            push!(valid_idx, k)
        end
    end
    return valid_idx
end


"""
    square_indices(i::Int, j::Int, L::Int)

Return all Cartesian indices in a square of side length `L` centered at `(i,j)`.

If `L` is odd, `(i,j)` is exactly at the center.
If `L` is even, the square is centered with `(i,j)` near the middle.
"""
function square_indices(cartesianIndex, L::Int)
    i,j = cartesianIndex[1],cartesianIndex[2]
    half = div(L, 2)
    range_i = (i - half):(i + half)
    range_j = (j - half):(j + half)
    return [ CartesianIndex(x,y) for x in range_i, y in range_j]
end


"""
    mean_grid(grid)

Return a grid with the mean of each list in `grid`.
If a list is empty, return 0.
"""
function mean_grid2(grid::AbstractArray{<:AbstractVector})
    return [length(v)<=1 ? v[1] : mean(v[1:end]) for v in grid]
end



# # ================================================================
# # K / ε-nearest patch extraction (SIGMA-BASED THRESHOLDING)
# # ================================================================
# function get_K_NearestPatches_NEW(
#     xPSM::AbstractMatrix,
#     X::AbstractMatrix,
#     Y::AbstractMatrix,
#     ref_Ind::Int,
#     TextutedPix_Ind,
#     patchSize::Int,
#     sigma_e::Float64,         # Explicit noise estimate (Standard Deviation)
#     search_radius::Int = 39,  # Search window radius
#     q::Float64 = 2.5,          # Threshold multiplier (lambda)
#     )

#     # 1. Setup and Reference Point
#     TextutedPix_Ind_Tup = [(e[1], e[2]) for e in TextutedPix_Ind]
#     hSize = patchSize ÷ 2
#     np_X, np_Y = size(xPSM)
#     ref_p = TextutedPix_Ind_Tup[ref_Ind]

#     # Pre-calculate reference views
#     r_range = ref_p[1]-hSize:ref_p[1]+hSize
#     c_range = ref_p[2]-hSize:ref_p[2]+hSize
#     ref_patch = @views xPSM[r_range, c_range]

#     # 2. Define BM3D Thresholding Logic
#     # Standard BM3D threshold: distance < lambda * sigma^2 * N_pixels
#     # This accounts for the fact that 'dist_sq' is a sum, not a mean.
#     d_threshold = q * (sigma_e^2) * (patchSize^2) 

#     candidate_distances = Float64[]
#     candidate_coords = Tuple{Int, Int}[]

#     # 3. Efficient Search (O(N) iteration instead of set intersection)
#     for (i, j) in TextutedPix_Ind_Tup
#         # Spatial constraint: Only look within search_radius of reference point
#         if abs(i - ref_p[1]) <= search_radius && abs(j - ref_p[2]) <= search_radius
            
#             # Boundary check for the candidate patch
#             if i > hSize && i <= np_X - hSize && j > hSize && j <= np_Y - hSize
                
#                 # Manual L2 Distance (Allocation-free sum of squared differences)
#                 dist_sq = 0.0
#                 @inbounds for dc in 1:patchSize, dr in 1:patchSize
#                     diff = ref_patch[dr, dc] - xPSM[i-hSize+dr-1, j-hSize+dc-1]
#                     dist_sq += diff * diff
#                 end

#                 # Inclusion based on noise-scaled distance
#                 if dist_sq <= d_threshold
#                     push!(candidate_distances, dist_sq)
#                     push!(candidate_coords, (i, j))
#                 end
#             end
#         end
#     end

#     # 4. Sorting and Result Extraction
#     p = sortperm(candidate_distances)
    
#     # Cap the number of patches to patchSize^2 (as per your original constraint)
#     # BM3D practitioners often use a power of 2 (8, 16, or 32) here.
#     max_k = min(length(p), patchSize^2)
#     keep_idx = p[1:max_k]

#     final_distances = candidate_distances[keep_idx]
#     final_coords = candidate_coords[keep_idx]

#     # Materialize the list of patches (Vector of Matrices)
#     closest_patches = [
#         Matrix(@views xPSM[c[1]-hSize:c[1]+hSize, c[2]-hSize:c[2]+hSize]) 
#         for c in final_coords
#     ]

#     return ref_p,
#            Matrix(ref_patch),
#            Matrix(@views X[r_range, c_range]),
#            Matrix(@views Y[r_range, c_range]),
#            final_distances,
#            closest_patches,
#            final_coords
# end

# # --- Updated Helper Function ---
# function combine_grids(E_grid, V_grid, method="ivw")
#     Nx, Ny = size(E_grid)
#     E = zeros(Float64, Nx, Ny)
#     VE = zeros(Float64, Nx, Ny)

#     for i in 1:Nx
#         for j in 1:Ny
#             est_list = E_grid[i, j]
#             var_list = V_grid[i, j]

#             if !isempty(var_list)
#                 if method == "ivw"
#                     weights = 1.0 ./ (var_list .+ 1e-12) 
#                     sum_weights = sum(weights)
#                     E[i, j] = sum(weights .* est_list) / sum_weights
#                     VE[i, j] = 1.0 / sum_weights
#                 elseif method == "min_var"
#                     min_idx = argmin(var_list)
#                     E[i, j] = est_list[min_idx]
#                     VE[i, j] = var_list[min_idx]
#                 end
#             else
#                 E[i, j], VE[i, j] = NaN, NaN
#             end
#         end
#     end
#     return E, VE
# end

function combine_grids(E_grid, V_grid, Y, method="ivw_skip_first"; tol=1e-8)
    Nx, Ny = size(E_grid)
    E = zeros(Float64, Nx, Ny)
    VE = zeros(Float64, Nx, Ny)

    for i in 1:Nx
        for j in 1:Ny
            est_list = E_grid[i, j]
            var_list = V_grid[i, j]

            if !isempty(var_list)
                if method == "ivw"
                    weights = 1.0 ./ (var_list .+ 1e-12) 
                    sum_weights = sum(weights)
                    E[i, j] = sum(weights .* est_list) / sum_weights
                    VE[i, j] = 1.0 / sum_weights
                
                elseif method == "min_var"
                    min_idx = argmin(var_list)
                    E[i, j] = est_list[min_idx]
                    VE[i, j] = var_list[min_idx]

                elseif method == "ivw_skip_first"
                    if length(est_list) > 1
                        sub_est, sub_var = est_list[2:end], var_list[2:end]
                    else
                        sub_est, sub_var = est_list, var_list
                    end
                    weights = 1.0 ./ (sub_var .+ 1e-12)
                    sum_weights = sum(weights)
                    E[i, j] = sum(weights .* sub_est) / sum_weights
                    VE[i, j] = 1.0 / sum_weights

                elseif method == "ivw_conditional"
                    if length(var_list) > 1 && argmin(var_list) != 1
                        weights = 1.0 ./ (var_list .+ 1e-12)
                        sum_weights = sum(weights)
                        E[i, j] = sum(weights .* est_list) / sum_weights
                        VE[i, j] = 1.0 / sum_weights
                    else
                        E[i, j] = est_list[1]
                        VE[i, j] = var_list[1]
                    end
                end

                # --- New Logic: If E[i,j] ~ Y[i,j], replace with first element ---
                if isapprox(E[i, j], Y[i, j], atol=tol)
                    E[i, j] = est_list[1]
                    VE[i, j] = var_list[1] # Update variance to match the first estimate
                end
                # -----------------------------------------------------------------

            else
                E[i, j], VE[i, j] = NaN, NaN
            end
        end
    end
    return E, VE
end

##### NOISE LEVEL ESTIMATION
@everywhere function sigma_estimate(y::Vector{Float64};M::Int=4)
    n = length(y)
    L = M + 1
    Vs = [Int64[k == 0 ? 1.0 : i^k for i in 0:(L-1)] for k in 0:(M-1)]
    Vs = mapreduce(permutedims, vcat, Vs)
    V = nullspace(Vs)
    Z = [sum(V[j] * y[i+j-1] for j in 1:length(V)) for i in 1:(n-length(V)+1)]
    
    #V = [1,-4,6,-4,1]/sqrt(70)
    #Z = [sum(V * y[i+j-1]) for j in 1:length(V) for i in 1:(n-length(V)+1)]
    #print(Z)
    #Robust STDZ estimate
    U = abs.(Z)
    U2 = U.^2
    σ_e = sqrt(mean(U2))
    return σ_e
end