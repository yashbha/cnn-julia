using StatsBase
using Feather
using DataFrames
using Base
using Statistics
using Knet
#=export

ma,
Next_batch,
predict_x,
train_data,
get_weights,
_genotype,
_phenotype,
_individual,
_population,
get_mean_variance,
get_fitness,
selection_roulette,
crossover_rand_points,
clone,
distance,
selection,
crossover,
actual_rets,
evaluate=#

dF = Feather.read("data.feather")

function ma(x, wind::Int)
 len = length(x)
 y = Vector(undef,len)
 for i in 1:len
     lo = max(1, i - wind)
     hi = min(len, i + wind)

     y[i] = mean(x[lo:hi])

 end
 return y
end

function Next_batch(df;block_size=5,num_size_time=10)

 a=size(df,1)-(block_size* (num_size_time+1) )

 x=sample(1:a,1)
 x=x[1]
 covar=[]
 for i in 1:num_size_time
     dd=df[x:x+block_size,:]
     dataforcov=Matrix(dd)
     cc=cov(dataforcov)
     x->x+block_size
     push!(covar,cc)
 end
 #x-=block_size
 rrets=[]
 dataas=[]
 l1=[]

 for i in 1:size(df,2)
     av=ma(df[:,i],block_size)
     if x<0
         println(i,":",x,"  a:",a)
     end
     push!(dataas,(av[x],covar[end]))
     r=1

     for j in 1:block_size
         r*=1+(df[x+j,i]/100)
     end
     r-=1
     r*=100
     if r>1
         push!(rrets,1)
     elseif r<-1
         push!(rrets,-1)
     else
         push!(rrets,0)
     end
     push!(l1,r)
 end

 return (covar,rrets,l1,dataas)
end

function predict_x(df,block_size=5,num_size_time=10)
 x=size(df,1)-(block_size* (num_size_time+1) )

 covar=[]
 for i in 1:num_size_time
     dd=df[x:x+block_size,:]
     dataforcov=Matrix(dd)
     cc=cov(dataforcov)
     x=>x+block_size
     push!(covar,cc)
 end
 x-=block_size
 rrets=[]
 dataas=[]
 l1=[]

 for i in 1:size(df,2)
     av=ma(df[:,i],block_size)
     push!(dataas,(av[end],covar[end]))
     r=1

     for j in 1:block_size
         r*=1+(df[x+j,i]/100)
     end
     r-=1
     r*=100
     if r>1
         push!(rrets,1)
     elseif r<-1
         push!(rrets,-1)
     else
         push!(rrets,0)
     end
     push!(l1,r)
 end

 return (covar,rrets,l1,dataas)
end

function train_data(indices)
 println(indices)
 ddd=dF[:,indices]
 println(typeof(ddd))
 xx=Matrix(ddd)
 cc=cov(xx)

 size(dF,2)

 a=[]
 for i in 1:1500

     push!(a,Next_batch(ddd))
 end
 push!(a,predict_x(ddd))

 return a
end

function get_weights(ran)
 s=train_data(ran)
 s1=zeros(10,10,10,1501)
 sy=zeros(10,1501)
 x_tdata=[]
 y_tdata=[]
 for i in 1:1501 push!(x_tdata,s[i][1]) end

 for i in 1:1501
      for j in 1:10
         s1[:,:,j,i]=x_tdata[i][j][:][:]
      end
 end
 xx=s1

 for i in 1:1501 push!(y_tdata,s[i][2]) end
 for i in 1:1501
 sy[:,i]=y_tdata[:][i]
 end
 t_x=xx[:,:,:,1:1500]
 testx=xx[:,:,:,end]
 t_y=y_tdata[1:1500][:]

 return (t_x,t_y)
end

let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray : Array)) : at
end


##Model definition

#=
Initialization is from
He et al., 2015,
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
https://arxiv.org/abs/1502.01852
=#
kaiming(et, h, w, i, o) = et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)


function init_modell(;et=Float32)
    # Use bnparams() to initialize gammas and betas
    w = Any[
        kaiming(et, 2, 2, 10, 16),    bnparams(et, 16),
        kaiming(et, 2, 2, 16, 32),   bnparams(et, 32),
        kaiming(et, 2, 2, 32, 64),   bnparams(et, 64),
        xavier(et, 100, 1024), bnparams(et, 100),
        xavier(et, 10, 100),         zeros(et, 10, 1)
    ]
    # Initialize a moments object for each batchnorm
    m = Any[bnmoments() for i = 1:4]

    w = map(atype(), w)

    return w, m
end

function conv_layer(w, m, x; maxpool=true)
    o = conv4(w[1], x; padding=1)
    o = batchnorm(o, m, w[2])
    o = relu.(o)
    if maxpool; o=Knet.pool(o); end
    return o
end

function lin_layer(w, m, x)
    o = w[1] * x
    o = batchnorm(o, m, w[2])
    return relu.(o)
end

function predict(w, m, x)
    #println("x",size(x))
    #println("w1",size(w[1]))
    o = conv_layer(w[1:2] , m[1], x)
    #println("o1",size(o))
    #println("w3",size(w[3]))
    o = conv_layer(w[3:4] , m[2], o)
    #println("o2",size(o))
    #println("w5",size(w[5]))
    o = conv_layer(w[5:6] , m[3], o; maxpool=false)
    #println("o3",size(o))
    #println("w7",size(w[7]))
    o = lin_layer( w[7:8] , m[4], mat(o))
    #println("o4",size(o))
    #println("w9",size(w[9]))

    return sigm.(w[9] * o .+ w[10])
end

function loss(w, m, x, classes)
    #println("w: ",size(w),"   x:  ",size(x))
    ypred = predict(w,m, x)
    #println("ypred:",size(ypred))
    #println("y:",size(classes))
    jl=nll(ypred, classes)
    #println("jl",jl)
    return jl
end

lossgrad = grad(loss)

# Training
function epoch!(w, m, o, xtrn, ytrn;  mbatch=64)
    data = minibatch(xtrn, ytrn, mbatch;
                   shuffle=true,
                   xtype=atype())
    for (x, y) in data
        g = lossgrad(w, m, x, y)
        update!(w, g, o)
    end
end

# Accuracy computation
function acc(w, m, xtst, ytst; mbatch=64)
    data = minibatch(xtst, ytst, mbatch;
                     partial=true,
                     xtype=atype())
    model = (w, m)
    return accuracy(model, data,
                    (model, x)->predict(model[1], model[2], x);
                    average=true)
end

# TODO: add command line options
function train(;optim=Momentum, epochs=5,
               lr=0.01, oparams...)
    w, m = init_modell()

    o = map(_->Momentum(;lr=lr, oparams...), w)
    (xtrn, ytrn) = get_weights(sample(1:202,10,replace=false))
    xtrn=convert(Array{Float32},xtrn)
    #println("This is w : ",size(w[1]))
    #println("This is x : ",size(xtrn))
    #ytrn=convert(Array{Float32},ytrn)
    for epoch = 1:epochs
        println("epoch: ", epoch)
        epoch!(w, m, o, xtrn, ytrn)
        println("train accuracy: ", acc(w, m, xtrn, ytrn))
        #println("test accuracy: ", acc(w, m, xtst, ytst))
    end
end
train()
