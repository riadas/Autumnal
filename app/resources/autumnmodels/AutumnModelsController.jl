module AutumnModelsController 
using Genie.Renderer
using Genie.Renderer.Html: html
using Genie.Renderer.Json: json
using Genie.Router 
using Genie.Requests
using MLStyle
using Random
using MacroTools: striplines
using DataStructures
using Statistics: median
using SExpressions
using Distributions: Categorical
using JLD 
using Dates

MODS = Dict{Int, Any}(); 
HISTORY = Dict{Int, Any}()
EVENTS = Dict{Int, Any}()

function autumnmodels()
  redirect(:playground)
end

function playground()
  html(:autumnmodels, :autumnmodelsdashboard, size=[0:255;])
end

function random()
  content = generatescene_program()
  json([content])
end

function random2()
  content = generateprogram()
  json([content])
end

function random3()
  content = generateprogram(group=true)
  json([content])
end

function compileautumn()
  # println("compileautumn")
  autumnString = postpayload(:autumnstring, "(program (= GRID_SIZE 12))")
  clientid = parse(Int64, postpayload(:clientid, 0))
  # println(string("autumnstring: ", autumnString))
  # println(string("clientid: ", clientid))
  # println(typeof(clientid))
  if !haskey(MODS, clientid)
    try
      parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
      content = autumnString
      MODS[clientid] = parsedAutumn
      @show typeof(MODS[clientid])
      # println("HERE 3")
    catch y
      # println("PARSING OR COMPILING FAILURE!")
      # println(y)
      content = ""
    end
  else
    # println("RESET")
    content = autumnString
    delete!(MODS, clientid)
  end
  json([content])
end

function step()
  # println("click")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  env_ = step(aex, env, empty_env())
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "nothing")

  json(vcat(background, cells))
  #json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].next(MODS[clientid].Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))))) : []))
end

function startautumn()
  # println("startautumn")
  clientid = parse(Int64, @params(:clientid))
  new_aex = MODS[clientid]
  if !(new_aex isa AExpr) 
    new_aex = new_aex[1]
  end
  println("HERE")
  @show typeof(new_aex)
  new_aex, env_ = start(new_aex)
  MODS[clientid] = (new_aex, env_)
  HISTORY[clientid] = []
  EVENTS[clientid] = []

  # println(state.time)
  grid_size = env_.GRID_SIZE isa AbstractArray ? [env_.GRID_SIZE] : [[env_.GRID_SIZE, env_.GRID_SIZE]]
  background = env_.state.scene.background
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  # push!(EVENTS[clientid], "nothing")

  json(vcat(grid_size, background, cells))
  # json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].init(nothing)) : []))
end

function clicked()
  # println("click")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  x = @params(:x)
  y = @params(:y)
  env_ = step(aex, env, (click=Click(parse(Int64, x), parse(Int64, y)),))
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "click $(x) $(y)")

  json(vcat(background, cells))
  #json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].next(MODS[clientid].Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))))) : []))
end

function up()
  # println("up")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  env_ = step(aex, env, (up=true,))
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "up")

  json(vcat(background, cells))
end

function down()
  # println("up")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  env_ = step(aex, env, (down=true,))
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "down")

  json(vcat(background, cells))
end

function right()
  # println("up")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  env_ = step(aex, env, (right=true,))
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "right")

  json(vcat(background, cells))
end

function left()
  # println("up")
  clientid = parse(Int64, @params(:clientid))
  aex, env = MODS[clientid]
  env_ = step(aex, env, (left=true,))
  MODS[clientid] = (aex, env_)
  background = env_.state.scene.background  
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :render, env_.state.scene), env_)[1])
  push!(HISTORY[clientid], vcat(background, cells))
  push!(EVENTS[clientid], "left")

  json(vcat(background, cells))
end

function replay()
  # println("replay")
  clientid = parse(Int64, @params(:clientid))
  json(HISTORY[clientid])
  # json(Dict(key => map(particle -> [particle.position.x, particle.position.y, particle.color], filter(particle -> particle.render, particles)) for (key, particles) in history))
end

# function save()
#   log_dir = "/Users/riadas/Documents/urop/today_temp/CausalDiscovery.jl/logged_observations"
#   clientid = parse(Int64, @params(:clientid))
#   curr_time = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
#   JLD.save("$(log_dir)/time_$(curr_time)_client_$(clientid).jld", "observations", HISTORY[clientid])
#   JLD.save("$(log_dir)/time_$(curr_time)_client_$(clientid)_EVENTS.jld", "observations", EVENTS[clientid])
#   open("$(log_dir)/time_$(curr_time)_client_$(clientid)_EVENTS.txt","w") do io
#     println(io, join(EVENTS[clientid], "\n"))
#   end
  
#   json([])
# end

const log_dir = "/Users/riadas/Documents/urop/today_temp/CausalDiscovery.jl/saved/"

function save()
  # get params 
  clientid = parse(Int64, @params(:clientid))
  model_name = @params(:model)
  grid_size = parse(Int64, @params(:gridsize))

  # create model directory in saved/
  directory = string(log_dir, model_name)
  if !isdir(directory)
    mkdir(directory)
  end

  # create data dict 
  data = Dict(["observations" => HISTORY[clientid], 
               "user_events" => EVENTS[clientid],
               "grid_size" => grid_size])

  # save file 
  index = 1 + length(filter(x -> occursin(".jld", x), readdir(directory)))
  file_path = string(directory, "/", index, ".jld")
  JLD.save(file_path, data)
  
  json([])
end

function synthesize()
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  grid_size = state.GRID_SIZEHistory[state.time]
  observations = [HISTORY[clientid][i] for i in 1:(min(20, length(HISTORY[clientid])))]
  @show observations
  @show grid_size
  content = singletimestepsolution_program(observations, grid_size)
  json([content])
end

# aexpr.jl


"Autumn Expression"
struct AExpr
  head::Symbol
  args::Vector{Any}
  AExpr(head::Symbol, @nospecialize args...) = new(head, [args...])
end
"Arguements of expression"
function args end

args(aex::AExpr) = aex.args
head(aex::AExpr) = aex.head
args(ex::Expr) = ex.args

"Expr in ith location in arg"
arg(aex, i) = args(aex)[i]

Base.Expr(aex::AExpr) = Expr(aex.head, aex.args...)

# Expression types
"Is `sym` a type symbol"
istypesymbol(sym) = (q = string(sym); length(q) > 0 && isuppercase(q[1]))
istypevarsymbol(sym) = (q = string(sym); length(q) > 0 && islowercase(q[1]))

# ## Printing
isinfix(f::Symbol) = f ∈ [:+, :-, :/, :*, :&&, :||, :>=, :<=, :>, :<, :(==)]
isinfix(f) = false


"Pretty print"
function showstring(expr::Expr)
  @match expr begin
    Expr(:program, statements...) => join(map(showstring, expr.args), "\n")
    Expr(:producttype, ts) => join(map(showstring, ts), "×")
    Expr(:functiontype, int, outt) => "($(showstring(int)) -> $(showstring(outt)))"
    Expr(:functiontype, vars...) => reduce(((x, y) -> string(showstring(x), " -> ", showstring(y))), vars)
    Expr(:typedecl, x, val) => "$x : $(showstring(val))"
    Expr(:externaldecl, x, val) => "external $x : $(showstring(val))"
    Expr(:external, val) => "external $(showstring(val))"
    Expr(:assign, x, val) => "$x $(needequals(val)) $(showstring(val))"
    Expr(:if, i, t, e) => "if ($(showstring(i)))\n  then ($(showstring(t)))\n  else ($(showstring(e)))"
    Expr(:initnext, i, n) => "init $(showstring(i)) next $(showstring(n))"
    Expr(:args, args...) => join(map(showstring, args), " ")
    Expr(:call, f, arg1, arg2) && if isinfix(f) end => "$(showstring(arg1)) $f $(showstring(arg2))"
    Expr(:call, f, args...) => "($(join(map(showstring, [f ; args]), " ")))"
    Expr(:let, vars...) => "let \n\t$(join(map(showstring, vars), "\n\t"))"
    Expr(:paramtype, type, param) => string(type, " ", param)
    Expr(:paramtype, type) => string(type)
    Expr(:case, type, vars...) => string("\n\tcase $(showstring(type)) of \n\t\t", join(map(showstring, vars), "\n\t\t"))
    Expr(:casevalue, type, value) => "$(showstring(type)) => $value"
    Expr(:casetype, type, vars...) => "$type $(join(vars, " "))"
    Expr(:type, vars...) => "type $(vars[1]) $(vars[2]) = $(join(map(showstring, vars[3:length(vars)]), " | "))"
    Expr(:typealias, var, val) => "type alias $(showstring(var)) = $(showstring(val))"
    Expr(:fn, params, body) => "fn $(showstring(params)) ($(showstring(body)))"
    Expr(:list, vals...) => "($(join(vals, ", ")))"
    Expr(:field, var, field) => "$(showstring(var)).$(showstring(field))"
    Expr(:typealiasargs, vals...) => "$(string("{ ", join(map(showstring, vals), ", ")," }"))"
    Expr(:lambda, var, val) => "($(showstring(var)) -> ($(showstring(val))))"
    Expr(:object, name, args...) => "object $(showstring(name)) {$(join(map(showstring, args), ","))}"
    Expr(:on, name, args...) => "on $(showstring(name)) ($(join(map(showstring, args))))"
    x                       => "Fail $x"

  end
end

showstring(lst::Array{}) = join(map(string, lst), " ")
showstring(str::String) = str

function needequals(val)
  if typeof(val) == Expr && val.head == :fn
    ""
   else
    "="
  end
end

showstring(aexpr::AExpr) = showstring(Expr(aexpr))

showstring(s::Union{Symbol, Integer}) = s
showstring(s::Type{T}) where {T <: Number} = s
Base.show(io::IO, aexpr::AExpr) = print(io, showstring(aexpr))


# sexpr.jl
fg(s) = s
fg(s::SExpressions.Cons) = array(s)
"Convert an `SExpression` into nested Array{Any}"
array(s::SExpression) = [map(fg, s)...]

@inline rest(sexpr::SExpressions.Cons) = sexpr.cdr
@inline rest2(sexpr::SExpressions.Cons) = rest(rest(sexpr))
"""Parse string `saexpr` into AExpr

```julia

prog = \"\"\"
(program
  (external (:: x Int))
  (:: y Float64)
  (group Thing (:: position Int) (:: alpha Bool))
  (= y 1.2)
  (= 
    (-> (x y)
        (let (z (+ x y))
              (* z y)))
)
\"\"\"

"""
parseautumn(sexprstring::AbstractString) =
  parseau(array(SExpressions.Parser.parse(sexprstring)))

"Parse SExpression into Autumn Expressions"
function parseau(sexpr::AbstractArray)
  res = MLStyle.@match sexpr begin
    [:program, lines...]              => AExpr(:program, map(parseau, lines)...)
    [:if, c, :then, t, :else, e]      => AExpr(:if, parseau(c), parseau(t), parseau(e))
    [:initnext, i, n]                 => AExpr(:initnext, parseau(i), parseau(n))
    [:(=), x::Symbol, y]              => AExpr(:assign, x, parseau(y))
    [:(:), v::Symbol, τ]              => AExpr(:typedecl, v, parsetypeau(τ))
    [:external, tdef]                 => AExpr(:external, parseau(tdef))
    [:let, vars]                      => AExpr(:let, map(parseau, vars)...)
    [:case, name, cases...]           => AExpr(:case, name, map(parseau, cases)...)
    [:(=>), type, value]              => AExpr(:casevalue, parseau(type), parseau(value))
    [:type, :alias, var, val]         => AExpr(:typealias, var, parsealias(val))
    [:fn, params, body]               => AExpr(:fn, AExpr(:list, params...), parseau(body))
    [:(-->), var, val]                => AExpr(:lambda, parseau(var), parseau(val))
    [:list, vars...]                  => AExpr(:list, map(parseau, vars)...)
    [:.., var, field]                 => AExpr(:field, parseau(var), parseau(field))
    [:on, args...]                    => AExpr(:on, map(parseau, args)...)
    [:object, args...]                => AExpr(:object, map(parseau, args)...)
    [f, xs...]                        => AExpr(:call, parseau(f), map(parseau, xs)...)
    [vars...]                         => AExpr(:list, map(parseau, vars)...)
  end
end

function parsealias(expr)
  AExpr(:typealiasargs, map(parseau, expr)...)
end

#(: map (-> (-> a b) (List a) (List b)))
function parsetypeau(sexpr::AbstractArray)
  MLStyle.@match sexpr begin
    [τ, tvs...] && if (istypesymbol(τ) && all(istypevarsymbol.(tvs)))   end => AExpr(:paramtype, τ, tvs...)
    [:->, τs...]                                                            => AExpr(:functiontype, map(parsetypeau, τs)...)
    [args...]                                                               => [args...]
  end
end

parseau(list::Array{Int, 1}) = list[1]
parsetypeau(s::Symbol) = s
parseau(s::Symbol) = s
parseau(s::Union{Number, String}) = s

"""
Macro for parsing autumn
au\"\"\"
(program
  (= x 3)
  (let (x 3) (+ x 3))
)
\"\"\"
"""
macro au_str(x::String)
  QuoteNode(parseautumn(x))
end

# compileutils.jl

# binary operators
binaryOperators = map(string, [:+, :-, :/, :*, :&, :|, :>=, :<=, :>, :<, :(==), :!=, :%, :&&])

abstract type Object end
# ----- Compile Helper Functions ----- 

function compile(expr::AExpr, data::Dict{String, Any}, parent::Union{AExpr, Nothing}=nothing)::Expr
  arr = [expr.head, expr.args...]
  # print(arr)
  # print("\n")
  res = @match arr begin
    [:if, args...] => :($(compile(args[1], data)) ? $(compile(args[2], data)) : $(compile(args[3], data)))
    [:assign, args...] => compileassign(expr, data, parent)
    [:typedecl, args...] => compiletypedecl(expr, data, parent)
    [:external, args...] => compileexternal(expr, data)
    [:let, args...] => compilelet(expr, data)
    [:case, args...] => compilecase(expr, data)
    [:typealias, args...] => compiletypealias(expr, data)
    [:lambda, args...] => :($(compile(args[1], data)) -> $(compile(args[2], data)))
    [:list, args...] => :([$(map(x -> compile(x, data), expr.args)...)])
    [:call, args...] => compilecall(expr, data)
    [:field, args...] => :($(compile(expr.args[1], data)).$(compile(expr.args[2], data)))
    [:object, args...] => compileobject(expr, data)
    [:on, args...] => compileon(expr, data)
    [args...] => throw(AutumnError(string("Invalid AExpr Head: ", expr.head))) # if expr head is not one of the above, throw error
  end
end

function compile(expr::AbstractArray, data::Dict{String, Any}, parent::Union{AExpr, Nothing}=nothing)
  if length(expr) == 0 || (length(expr) > 1 && expr[1] != :List)
    throw(AutumnError("Invalid List Syntax"))
  elseif expr[1] == :List
    :(Array{$(compile(expr[2:end], data))})
  else
    compile(expr[1], data)      
  end
end

function compile(expr, data::Dict{String, Any}, parent::Union{AExpr, Nothing}=nothing)
  if expr isa BigInt
    floor(Int, expr)
  elseif expr == Symbol("true")
    :(1 == 1)
  elseif expr == Symbol("false")
    :(1 == 2)
  elseif expr in [:left, :right, :up, :down]
    :(occurred($(expr)))
  elseif expr == :clicked
    :(occurred(click))
  else
    expr
  end
end

function compileassign(expr::AExpr, data::Dict{String, Any}, parent::Union{AExpr, Nothing})
  # get type, if declared
  type = haskey(data["types"], expr.args[1]) ? data["types"][expr.args[1]] : nothing
  if (typeof(expr.args[2]) == AExpr && expr.args[2].head == :fn) # handle function assignments
    if type !== nothing # handle function with typed arguments/return type
      args = compile(expr.args[2].args[1], data).args # function args
      argtypes = map(x -> compile(x, data), type.args[1:(end-1)]) # function arg types
      tuples = [(args[i], argtypes[i]) for i in [1:length(args);]]
      typedargexprs = map(x -> :($(x[1])::$(x[2])), tuples)
      quote 
        function $(compile(expr.args[1], data))($(typedargexprs...))::$(compile(type.args[end], data))
          $(compile(expr.args[2].args[2], data))  
        end
      end 
    else # handle function without typed arguments/return type
      quote 
        function $(compile(expr.args[1], data))($(compile(expr.args[2].args[1], data).args...))
            $(compile(expr.args[2].args[2], data))
        end 
      end          
    end
  else # handle non-function assignments
    # handle global assignments
    if parent !== nothing && (parent.head == :program) 
      if (typeof(expr.args[2]) == AExpr && expr.args[2].head == :initnext)
        push!(data["initnext"], expr)
      else
        push!(data["lifted"], expr)
      end
      :()
    # handle non-global assignments
    else 
      if type !== nothing
        # :($(compile(expr.args[1], data))::$(compile(type, data)) = $(compile(expr.args[2], data)))
        :($(compile(expr.args[1], data)) = $(compile(expr.args[2], data)))
      else
        :($(compile(expr.args[1], data)) = $(compile(expr.args[2], data)))
      end
    end
  end
end

function compiletypedecl(expr::AExpr, data::Dict{String, Any}, parent::Union{AExpr, Nothing})
  if (parent !== nothing && (parent.head == :program || parent.head == :external))
    # # println(expr.args[1])
    # # println(expr.args[2])
    data["types"][expr.args[1]] = expr.args[2]
    :()
  else
    :(local $(compile(expr.args[1], data))::$(compile(expr.args[2], data)))
  end
end

function compileexternal(expr::AExpr, data::Dict{String, Any})
  # # println("here: ")
  # # println(expr.args[1])
  if !(expr.args[1] in data["external"])
    push!(data["external"], expr.args[1])
  end
  compiletypedecl(expr.args[1], data, expr)
end

function compiletypealias(expr::AExpr, data::Dict{String, Any})
  name = expr.args[1]
  fields = map(field -> (
    :($(field.args[1])::$(field.args[2]))
  ), expr.args[2].args)
  quote
    struct $(name)
      $(fields...) 
    end
  end
end

function compilecall(expr::AExpr, data::Dict{String, Any})
  fnName = expr.args[1]
  if fnName == :clicked
    :(clicked(click, $(map(x -> compile(x, data), expr.args[2:end])...)))
  elseif fnName == :uniformChoice
    :(uniformChoice(rng, $(map(x -> compile(x, data), expr.args[2:end])...)))
  elseif !(fnName in binaryOperators) && fnName != :prev
    :($(fnName)($(map(x -> compile(x, data), expr.args[2:end])...)))
  elseif fnName == :prev
    if expr.args[2] == :obj
      :($(fnName)($(map(x -> compile(x, data), expr.args[2:end])...)))
    else
      :($(Symbol(string(expr.args[2]) * "Prev"))($(map(compile, expr.args[3:end])...)))
    end
  elseif fnName != :(==)        
    :($(fnName)($(compile(expr.args[2], data)), $(compile(expr.args[3], data))))
  else
    :($(compile(expr.args[2], data)) == $(compile(expr.args[3], data)))
  end
end

function compilelet(expr::AExpr, data::Dict{String, Any})
  quote
    $(map(x -> compile(x, data), expr.args)...)
  end
end

function compilecase(expr::AExpr, data::Dict{String, Any})
  quote 
    @match $(compile(expr.args[1], data)) begin
      $(map(x -> :($(compile(x.args[1], data)) => $(compile(x.args[2], data))), expr.args[2:end])...)
    end
  end
end

function compileobject(expr::AExpr, data::Dict{String, Any})
  name = expr.args[1]
  # # println("NAME")
  # # println(name)
  # # println(expr)
  # # println(expr.args)
  push!(data["objects"], name)
  custom_fields = map(field -> (
    :($(field.args[1])::$(compile(field.args[2], data)))
  ), filter(x -> (typeof(x) == AExpr && x.head == :typedecl), expr.args[2:end]))
  custom_field_names = map(field -> field.args[1], filter(x -> (x isa AExpr && x.head == :typedecl), expr.args[2:end]))
  rendering = compile(filter(x -> (typeof(x) != AExpr) || (x.head != :typedecl), expr.args[2:end])[1], data)
  quote
    mutable struct $(name) <: Object
      id::Int
      origin::Position
      alive::Bool
      changed::Bool
      $(custom_fields...) 
      render::Array{Cell}
    end

    function $(name)($(vcat(custom_fields, :(origin::Position))...))::$(name)
      state.objectsCreated += 1
      rendering = $(rendering)      
      $(name)(state.objectsCreated, origin, true, false, $(custom_field_names...), rendering isa AbstractArray ? vcat(rendering...) : [rendering])
    end
  end
end

function compileon(expr::AExpr, data::Dict{String, Any})
  # # println("here")
  # # println(typeof(expr.args[1]) == AExpr ? expr.args[1].args[1] : expr.args[1])
  # event = compile(expr.args[1], data)
  # response = compile(expr.args[2], data)

  event = expr.args[1]
  response = expr.args[2]
  push!(data["on"], (event, response))
  :()
end

function compileinitnext(data::Dict{String, Any})
  # ensure changed field is set to false 
  list_objects = filter(x -> get(data["types"], x, :Any) in map(x -> [:List, x], data["objects"]), 
                        map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))
  init = quote
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), data["lifted"])...)
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2].args[1], data))), data["initnext"])...)
    $(map(x -> :(foreach(x -> x.changed = false, $(compile(x, data)))), list_objects)...)
  end

  onClauses = map(x -> quote 
    if $(compile(x[1], data))
      $(compile(x[2], data))
    end
  end, data["on"])

  notOnClausesConstant = map(x -> quote 
                          if !(foldl(|, [$(map(y -> compile(y[1], data), filter(z -> ((z[2].head == :assign) && (z[2].args[1] == x.args[1])) || ((z[2].head == :let) && (x.args[1] in map(w -> w.args[1], z[2].args))), data["on"]))...)]; init=false))
                            $(compile(x.args[1], data)) = $(compile(x.args[2].args[2], data));                                 
                          end
                        end, filter(x -> !(get(data["types"], x.args[1], :Any) in map(y -> [:List, y], data["objects"])), 
                        data["initnext"]))

  notOnClausesList = map(x -> quote 
                        $(Meta.parse(string(compile(x.args[1], data), "Changed"))) = filter(o -> o.changed, $(compile(x.args[1], data)))
                        $(compile(x.args[1], data)) = filter(o -> !(o.id in map(x -> x.id, $(Meta.parse(string(compile(x.args[1], data), "Changed"))))), $(compile(x.args[2].args[2], data)))
                        $(compile(x.args[1], data)) = vcat($(Meta.parse(string(compile(x.args[1], data), "Changed")))..., $(compile(x.args[1], data))...)
                        $(compile(x.args[1], data)) = filter(o -> o.alive, $(compile(x.args[1], data)))
                        foreach(o -> o.changed = false, $(compile(x.args[1], data)))
                      end, filter(x -> get(data["types"], x.args[1], :Any) in map(y -> [:List, y], data["objects"]), 
                      data["initnext"]))

  next = quote
    $(map(x -> :($(compile(x.args[1], data)) = state.$(Symbol(string(x.args[1])*"History"))[state.time - 1]), 
      vcat(data["initnext"], data["lifted"]))...)
    $(onClauses...)
    $(notOnClausesConstant...)
    $(notOnClausesList...)
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))...)
  end

  initFunction = quote
    function init($(map(x -> :($(x.args[1])::Union{$(compile(data["types"][x.args[1]], data)), Nothing}), data["external"])...), custom_rng=rng)::STATE
      global rng = custom_rng
      $(compileinitstate(data))
      $(init)
      $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))...)
      $(map(x -> :(state.$(Symbol(string(x.args[1])*"History"))[state.time] = $(x.args[1])), 
            vcat(data["external"], data["initnext"], data["lifted"]))...)
      state.scene = Scene(vcat([$(filter(x -> get(data["types"], x, :Any) in vcat(data["objects"], map(x -> [:List, x], data["objects"])), 
                                  map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))...)]...), :backgroundHistory in fieldnames(STATE) ? state.backgroundHistory[state.time] : "#ffffff00")
      
      global state = state
      state
    end
    end
  nextFunction = quote
    function next($([:(old_state::STATE), map(x -> :($(x.args[1])::Union{$(compile(data["types"][x.args[1]], data)), Nothing}), data["external"])...]...))::STATE
      global state = old_state
      state.time = state.time + 1
      $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] == :GRID_SIZE, data["lifted"]))...)
      $(next)
      $(map(x -> :(state.$(Symbol(string(x.args[1])*"History"))[state.time] = $(x.args[1])), 
            vcat(data["external"], data["initnext"], data["lifted"]))...)
      state.scene = Scene(vcat([$(filter(x -> get(data["types"], x, :Any) in vcat(data["objects"], map(x -> [:List, x], data["objects"])), 
        map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))...)]...), :backgroundHistory in fieldnames(STATE) ? state.backgroundHistory[state.time] : "#ffffff00")
      global state = state
      state
    end
    end
    [initFunction, nextFunction]
end

# construct STATE struct
function compilestatestruct(data::Dict{String, Any})
  stateParamsInternal = map(expr -> :($(Symbol(string(expr.args[1]) * "History"))::Dict{Int64, $(haskey(data["types"], expr.args[1]) ? compile(data["types"][expr.args[1]], data) : Any)}), 
                            vcat(data["initnext"], data["lifted"]))
  stateParamsExternal = map(expr -> :($(Symbol(string(expr.args[1]) * "History"))::Dict{Int64, Union{$(compile(data["types"][expr.args[1]], data)), Nothing}}), 
                            data["external"])
  quote
    mutable struct STATE
      time::Int
      objectsCreated::Int
      scene::Scene
      $(stateParamsInternal...)
      $(stateParamsExternal...)
    end
  end
end

# initialize state::STATE variable
function compileinitstate(data::Dict{String, Any})
  initStateParamsInternal = map(expr -> :(Dict{Int64, $(haskey(data["types"], expr.args[1]) ? compile(data["types"][expr.args[1]], data) : Any)}()), 
                                vcat(data["initnext"], data["lifted"]))
  initStateParamsExternal = map(expr -> :(Dict{Int64, Union{$(compile(data["types"][expr.args[1]], data)), Nothing}}()), 
                                data["external"])
  initStateParams = [0, 0, :(Scene([])), initStateParamsInternal..., initStateParamsExternal...]
  initState = :(state = STATE($(initStateParams...)))
  initState
end

# ----- Built-In and Prev Function Helpers ----- #

function compileprevfuncs(data::Dict{String, Any})
  prevFunctions = map(x -> quote
        function $(Symbol(string(x.args[1]) * "Prev"))(n::Int=1)::$(haskey(data["types"], x.args[1]) ? compile(data["types"][x.args[1]], data) : Any)
          state.$(Symbol(string(x.args[1]) * "History"))[state.time - n >= 0 ? state.time - n : 0] 
        end
        end, 
  vcat(data["initnext"], data["lifted"]))
  prevFunctions = vcat(prevFunctions, map(x -> quote
        function $(Symbol(string(x.args[1]) * "Prev"))(n::Int=1)::Union{$(compile(data["types"][x.args[1]], data)), Nothing}
          state.$(Symbol(string(x.args[1]) * "History"))[state.time - n >= 0 ? state.time - n : 0] 
        end
        end, 
  data["external"]))
  prevFunctions
end

function compilebuiltin()
  occurredFunction = builtInDict["occurred"]
  uniformChoiceFunction = builtInDict["uniformChoice"]
  uniformChoiceFunction2 = builtInDict["uniformChoice2"]
  minFunction = builtInDict["min"]
  rangeFunction = builtInDict["range"]
  utils = builtInDict["utils"]
  [occurredFunction, utils, uniformChoiceFunction, uniformChoiceFunction2, minFunction, rangeFunction]
end

const builtInDict = Dict([
"occurred"        =>  quote
                        function occurred(click)
                          !isnothing(click)
                        end
                      end,
"uniformChoice"   =>  quote
                        function uniformChoice(rng, freePositions)
                          freePositions[rand(rng, Categorical(ones(length(freePositions))/length(freePositions)))]
                        end
                      end,
"uniformChoice2"   =>  quote
                        function uniformChoice(rng, freePositions, n)
                          map(idx -> freePositions[idx], rand(rng, Categorical(ones(length(freePositions))/length(freePositions)), n))
                        end
                      end,
"min"              => quote
                        function min(arr)
                          min(arr...)
                        end
                      end,
"range"           => quote
                      function range(start::Int, stop::Int)
                        [start:stop;]
                      end
                    end,
"utils"           => quote
                        abstract type Object end
                        abstract type KeyPress end

                        struct Left <: KeyPress end
                        struct Right <: KeyPress end
                        struct Up <: KeyPress end
                        struct Down <: KeyPress end

                        struct Click
                          x::Int
                          y::Int                    
                        end     

                        struct Position
                          x::Int
                          y::Int
                        end

                        struct Cell 
                          position::Position
                          color::String
                          opacity::Float64
                        end

                        Cell(position::Position, color::String) = Cell(position, color, 0.8)
                        Cell(x::Int, y::Int, color::String) = Cell(Position(floor(Int, x), floor(Int, y)), color, 0.8)
                        Cell(x::Int, y::Int, color::String, opacity::Float64) = Cell(Position(floor(Int, x), floor(Int, y)), color, opacity)

                        struct Scene
                          objects::Array{Object}
                          background::String
                        end

                        Scene(objects::AbstractArray) = Scene(objects, "#ffffff00")

                        function prev(obj::Object)
                          prev_objects = filter(o -> o.id == obj.id, state.scene.objects)
                          if prev_objects != []
                            prev_objects[1]                            
                          else
                            obj
                          end
                        end

                        function render(scene::Scene)::Array{Cell}
                          vcat(map(obj -> render(obj), filter(obj -> obj.alive, scene.objects))...)
                        end

                        function render(obj::Object)::Array{Cell}
                          if obj.alive
                            map(cell -> Cell(move(cell.position, obj.origin), cell.color), obj.render)
                          else
                            []
                          end
                        end

                        function isWithinBounds(obj::Object)::Bool
                          # # println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
                          length(filter(cell -> !isWithinBounds(cell.position), render(obj))) == 0
                        end

                        function clicked(click::Union{Click, Nothing}, object::Object)::Bool
                          if click == nothing
                            false
                          else
                            GRID_SIZE = state.GRID_SIZEHistory[0]
                            if GRID_SIZE isa AbstractArray 
                              GRID_SIZE_X = GRID_SIZE[1]
                              GRID_SIZE_Y = GRID_SIZE[2]

                              nums = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(object))
                              (GRID_SIZE * click.y + click.x) in nums
                            else
                              GRID_SIZE_X = GRID_SIZE
                              GRID_SIZE_Y = GRID_SIZE
                              nums = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(object))
                              (GRID_SIZE_X * click.y + click.x) in nums
                            end
                          end
                        end

                        function clicked(click::Union{Click, Nothing}, objects::AbstractArray)
                          # # println("LOOK AT ME")
                          # # println(reduce(&, map(obj -> clicked(click, obj), objects)))
                          reduce(|, map(obj -> clicked(click, obj), objects))
                        end

                        function objClicked(click::Union{Click, Nothing}, objects::AbstractArray)::Union{Object, Nothing}
                          # println(click)
                          if isnothing(click)
                            nothing
                          else
                            clicked_objects = filter(obj -> clicked(click, obj), objects)
                            if length(clicked_objects) == 0
                              nothing
                            else
                              clicked_objects[1]
                            end
                          end

                        end

                        function clicked(click::Union{Click, Nothing}, x::Int, y::Int)::Bool
                          if click == nothing
                            false
                          else
                            click.x == x && click.y == y                         
                          end
                        end

                        function clicked(click::Union{Click, Nothing}, pos::Position)::Bool
                          if click == nothing
                            false
                          else
                            click.x == pos.x && click.y == pos.y                         
                          end
                        end

                        function intersects(obj1::Object, obj2::Object)::Bool
                          # println("INTERSECTS 1")
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1))
                            nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj2))
                            length(intersect(nums1, nums2)) != 0

                          else
                            nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
                            nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj2))
                            length(intersect(nums1, nums2)) != 0
                          end
                        end

                        function intersects(obj1::Object, obj2::Array{<:Object})::Bool
                          # println("INTERSECTS 2")
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1))
                            nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                            # println(length(intersect(nums1, nums2)) != 0)
                            length(intersect(nums1, nums2)) != 0
                          else
                            nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
                            nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                            # println(length(intersect(nums1, nums2)) != 0)
                            length(intersect(nums1, nums2)) != 0
                          end
                        end

                        function intersects(obj1::Array{<:Object}, obj2::Array{<:Object})::Bool
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj1)...))
                            nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                            # println(length(intersect(nums1, nums2)) != 0)
                            length(intersect(nums1, nums2)) != 0
                          else 
                            nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj1)...))
                            nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                            # println(length(intersect(nums1, nums2)) != 0)
                            length(intersect(nums1, nums2)) != 0
                          end
                        end

                        function intersects(list1, list2)::Bool
                          length(intersect(list1, list2)) != 0 
                        end

                        function intersects(object::Object)::Bool
                          objects = state.scene.objects
                          intersects(object, objects)
                        end

                        function addObj(list::Array{<:Object}, obj::Object)
                          obj.changed = true
                          new_list = vcat(list, obj)
                          new_list
                        end

                        function addObj(list::Array{<:Object}, objs::Array{<:Object})
                          foreach(obj -> obj.changed = true, objs)
                          new_list = vcat(list, objs)
                          new_list
                        end

                        function removeObj(list::Array{<:Object}, obj::Object)
                          new_list = deepcopy(list)
                          for x in filter(o -> o.id == obj.id, new_list)
                            x.alive = false 
                            x.changed = true
                          end
                          new_list
                        end
    
                        function removeObj(list::Array{<:Object}, fn)
                          new_list = deepcopy(list)
                          for x in filter(obj -> fn(obj), new_list)
                            x.alive = false 
                            x.changed = true
                          end
                          new_list
                        end
    
                        function removeObj(obj::Object)
                          new_obj = deepcopy(obj)
                          new_obj.alive = false
                          new_obj.changed = true
                          new_obj
                        end

                        function updateObj(obj::Object, field::String, value)
                          fields = fieldnames(typeof(obj))
                          custom_fields = fields[5:end-1]
                          origin_field = (fields[2],)

                          constructor_fields = (custom_fields..., origin_field...)
                          constructor_values = map(x -> x == Symbol(field) ? value : getproperty(obj, x), constructor_fields)

                          new_obj = typeof(obj)(constructor_values...)
                          setproperty!(new_obj, :id, obj.id)
                          setproperty!(new_obj, :alive, obj.alive)
                          setproperty!(new_obj, :changed, true)

                          setproperty!(new_obj, Symbol(field), value)
                          state.objectsCreated -= 1    
                          new_obj
                        end

                        function filter_fallback(obj::Object)
                          true
                        end

                        function updateObj(list::Array{<:Object}, map_fn, filter_fn=filter_fallback)
                          orig_list = filter(obj -> !filter_fn(obj), list)
                          filtered_list = filter(filter_fn, list)
                          new_filtered_list = map(map_fn, filtered_list)
                          foreach(obj -> obj.changed = true, new_filtered_list)
                          vcat(orig_list, new_filtered_list)
                        end

                        function adjPositions(position::Position)::Array{Position}
                          filter(isWithinBounds, [Position(position.x, position.y + 1), Position(position.x, position.y - 1), Position(position.x + 1, position.y), Position(position.x - 1, position.y)])
                        end

                        function isWithinBounds(position::Position)::Bool
                          GRID_SIZE = state.GRID_SIZEHistory[0] 
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                            (position.x >= 0) && (position.x < GRID_SIZE_X) && (position.y >= 0) && (position.y < GRID_SIZE_Y)                          
                          else
                            (position.x >= 0) && (position.x < state.GRID_SIZEHistory[0]) && (position.y >= 0) && (position.y < state.GRID_SIZEHistory[0])                          
                          end
                        end

                        function isFree(position::Position)::Bool
                          length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, render(state.scene))) == 0
                        end

                        function isFree(click::Union{Click, Nothing})::Bool
                          if click == nothing
                            false
                          else
                            isFree(Position(click.x, click.y))
                          end
                        end

                        function rect(pos1::Position, pos2::Position)
                          min_x = pos1.x 
                          max_x = pos2.x 
                          min_y = pos1.y
                          max_y = pos2.y 
    
                          positions = []
                          for x in min_x:max_x 
                            for y in min_y:max_y
                              push!(positions, Position(x, y))
                            end
                          end
                          positions
                        end

                        function unitVector(position1::Position, position2::Position)::Position
                          deltaX = position2.x - position1.x
                          deltaY = position2.y - position1.y
                          if (floor(Int, abs(sign(deltaX))) == 1 && floor(Int, abs(sign(deltaY))) == 1)
                            Position(sign(deltaX), 0)
                            # uniformChoice(rng, [Position(sign(deltaX), 0), Position(0, sign(deltaY))])
                          else
                            Position(sign(deltaX), sign(deltaY))  
                          end
                        end

                        function unitVector(object1::Object, object2::Object)::Position
                          position1 = object1.origin
                          position2 = object2.origin
                          unitVector(position1, position2)
                        end

                        function unitVector(object::Object, position::Position)::Position
                          unitVector(object.origin, position)
                        end

                        function unitVector(position::Position, object::Object)::Position
                          unitVector(position, object.origin)
                        end

                        function unitVector(position::Position)::Position
                          unitVector(Position(0,0), position)
                        end 

                        function displacement(position1::Position, position2::Position)::Position
                          Position(floor(Int, position2.x - position1.x), floor(Int, position2.y - position1.y))
                        end

                        function displacement(cell1::Cell, cell2::Cell)::Position
                          displacement(cell1.position, cell2.position)
                        end

                        function adjacent(position1::Position, position2::Position):Bool
                          displacement(position1, position2) in [Position(0,1), Position(1, 0), Position(0, -1), Position(-1, 0)]
                        end

                        function adjacent(cell1::Cell, cell2::Cell)::Bool
                          adjacent(cell1.position, cell2.position)
                        end

                        function adjacent(cell::Cell, cells::Array{Cell})
                          length(filter(x -> adjacent(cell, x), cells)) != 0
                        end

                        function adjacentObjs(obj::Object)
                          filter(o -> adjacent(o.origin, obj.origin) && (obj.id != o.id), state.scene.objects)
                        end
    
                        function adjacentObjsDiag(obj::Object)
                          filter(o -> adjacentDiag(o.origin, obj.origin) && (obj.id != o.id), state.scene.objects)
                        end
    
                        function adjacentDiag(position1::Position, position2::Position)
                          displacement(position1, position2) in [Position(0,1), Position(1, 0), Position(0, -1), Position(-1, 0),
                                                                 Position(1,1), Position(1, -1), Position(-1, 1), Position(-1, -1)]
                        end

                        function rotate(object::Object)::Object
                          new_object = deepcopy(object)
                          new_object.render = map(x -> Cell(rotate(x.position), x.color), new_object.render)
                          new_object
                        end

                        function rotate(position::Position)::Position
                          Position(-position.y, position.x)
                         end

                        function rotateNoCollision(object::Object)::Object
                          (isWithinBounds(rotate(object)) && isFree(rotate(object), object)) ? rotate(object) : object
                        end

                        function move(position1::Position, position2::Position)
                          Position(position1.x + position2.x, position1.y + position2.y)
                        end

                        function move(position::Position, cell::Cell)
                          Position(position.x + cell.position.x, position.y + cell.position.y)
                        end

                        function move(cell::Cell, position::Position)
                          Position(position.x + cell.position.x, position.y + cell.position.y)
                        end

                        function move(object::Object, position::Position)
                          new_object = deepcopy(object)
                          new_object.origin = move(object.origin, position)
                          new_object
                        end

                        function move(object::Object, x::Int, y::Int)::Object
                          move(object, Position(x, y))                          
                        end

                        # ----- begin left/right move ----- #

                        function moveLeft(object::Object)::Object
                          move(object, Position(-1, 0))                          
                        end

                        function moveRight(object::Object)::Object
                          move(object, Position(1, 0))                          
                        end

                        function moveUp(object::Object)::Object
                          move(object, Position(0, -1))                          
                        end

                        function moveDown(object::Object)::Object
                          move(object, Position(0, 1))                          
                        end

                        # ----- end left/right move ----- #

                        function moveNoCollision(object::Object, position::Position)::Object
                          (isWithinBounds(move(object, position)) && isFree(move(object, position.x, position.y), object)) ? move(object, position.x, position.y) : object 
                        end

                        function moveNoCollision(object::Object, x::Int, y::Int)
                          (isWithinBounds(move(object, x, y)) && isFree(move(object, x, y), object)) ? move(object, x, y) : object 
                        end

                        # ----- begin left/right moveNoCollision ----- #

                        function moveLeftNoCollision(object::Object)::Object
                          (isWithinBounds(move(object, -1, 0)) && isFree(move(object, -1, 0), object)) ? move(object, -1, 0) : object 
                        end

                        function moveRightNoCollision(object::Object)::Object
                          (isWithinBounds(move(object, 1, 0)) && isFree(move(object, 1, 0), object)) ? move(object, 1, 0) : object 
                        end

                        function moveUpNoCollision(object::Object)::Object
                          (isWithinBounds(move(object, 0, -1)) && isFree(move(object, 0, -1), object)) ? move(object, 0, -1) : object 
                        end
                        
                        function moveDownNoCollision(object::Object)::Object
                          (isWithinBounds(move(object, 0, 1)) && isFree(move(object, 0, 1), object)) ? move(object, 0, 1) : object 
                        end

                        # ----- end left/right moveNoCollision ----- #

                        function moveWrap(object::Object, position::Position)::Object
                          new_object = deepcopy(object)
                          new_object.position = moveWrap(object.origin, position.x, position.y)
                          new_object
                        end

                        function moveWrap(cell::Cell, position::Position)
                          moveWrap(cell.position, position.x, position.y)
                        end

                        function moveWrap(position::Position, cell::Cell)
                          moveWrap(cell.position, position)
                        end

                        function moveWrap(object::Object, x::Int, y::Int)::Object
                          new_object = deepcopy(object)
                          new_object.position = moveWrap(object.origin, x, y)
                          new_object
                        end
                        
                        function moveWrap(position1::Position, position2::Position)::Position
                          moveWrap(position1, position2.x, position2.y)
                        end

                        function moveWrap(position::Position, x::Int, y::Int)::Position
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray  
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                            Position((position.x + x + GRID_SIZE_X) % GRID_SIZE_X, (position.y + y + GRID_SIZE_Y) % GRID_SIZE_Y)
                          else
                            # # println("hello")
                            # # println(Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE))
                            Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE)
                          end
                        end

                        # ----- begin left/right moveWrap ----- #

                        function moveLeftWrap(object::Object)::Object
                          new_object = deepcopy(object)
                          new_object.origin = moveWrap(object.origin, -1, 0)
                          new_object
                        end
                          
                        function moveRightWrap(object::Object)::Object
                          new_object = deepcopy(object)
                          new_object.origin = moveWrap(object.origin, 1, 0)
                          new_object
                        end

                        function moveUpWrap(object::Object)::Object
                          new_object = deepcopy(object)
                          new_object.origin = moveWrap(object.origin, 0, -1)
                          new_object
                        end

                        function moveDownWrap(object::Object)::Object
                          new_object = deepcopy(object)
                          new_object.origin = moveWrap(object.origin, 0, 1)
                          new_object
                        end

                        # ----- end left/right moveWrap ----- #

                        function randomPositions(GRID_SIZE, n::Int)::Array{Position}
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                            nums = uniformChoice(rng, [0:(GRID_SIZE_X * GRID_SIZE_Y - 1);], n)
                            # # println(nums)
                            # # println(map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums))
                            map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)
                          else
                            nums = uniformChoice(rng, [0:(GRID_SIZE * GRID_SIZE - 1);], n)
                            # # println(nums)
                            # # println(map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums))
                            map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
                          end
                        end

                        function distance(position1::Position, position2::Position)::Int
                          abs(position1.x - position2.x) + abs(position1.y - position2.y)
                        end

                        function distance(object1::Object, object2::Object)::Int
                          position1 = object1.origin
                          position2 = object2.origin
                          distance(position1, position2)
                        end

                        function distance(object::Object, position::Position)::Int
                          distance(object.origin, position)
                        end

                        function distance(position::Position, object::Object)::Int
                          distance(object.origin, position)
                        end

                        function closest(object::Object, type::DataType)::Position
                          objects_of_type = filter(obj -> (obj isa type) && (obj.alive), state.scene.objects)
                          if length(objects_of_type) == 0
                            object.origin
                          else
                            min_distance = min(map(obj -> distance(object, obj), objects_of_type))
                            filter(obj -> distance(object, obj) == min_distance, objects_of_type)[1].origin
                          end
                        end

                        function mapPositions(constructor, GRID_SIZE, filterFunction, args...)::Union{Object, Array{<:Object}}
                          map(pos -> constructor(args..., pos), filter(filterFunction, allPositions(GRID_SIZE)))
                        end

                        function allPositions(GRID_SIZE)
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                            nums = [0:(GRID_SIZE_X * GRID_SIZE_Y - 1);]
                            map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)
                          else 
                            nums = [0:(GRID_SIZE * GRID_SIZE - 1);]
                            map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
                          end
                        end

                        function updateOrigin(object::Object, new_origin::Position)::Object
                          new_object = deepcopy(object)
                          new_object.origin = new_origin
                          new_object
                        end

                        function updateAlive(object::Object, new_alive::Bool)::Object
                          new_object = deepcopy(object)
                          new_object.alive = new_alive
                          new_object
                        end

                        function nextLiquid(object::Object)::Object 
                          # # println("nextLiquid")                          
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                          else
                            GRID_SIZE_X = GRID_SIZE
                            GRID_SIZE_Y = GRID_SIZE
                          end
                          new_object = deepcopy(object)
                          if object.origin.y != GRID_SIZE_Y - 1 && isFree(move(object.origin, Position(0, 1)))
                            new_object.origin = move(object.origin, Position(0, 1))
                          else
                            leftHoles = filter(pos -> (pos.y == object.origin.y + 1)
                                                       && (pos.x < object.origin.x)
                                                       && isFree(pos), allPositions())
                            rightHoles = filter(pos -> (pos.y == object.origin.y + 1)
                                                       && (pos.x > object.origin.x)
                                                       && isFree(pos), allPositions())
                            if (length(leftHoles) != 0) || (length(rightHoles) != 0)
                              if (length(leftHoles) == 0)
                                closestHole = closest(object, rightHoles)
                                if isFree(move(closestHole, Position(0, -1)), move(object.origin, Position(1, 0)))
                                  new_object.origin = move(object.origin, unitVector(object, move(closestHole, Position(0, -1))))
                                end
                              elseif (length(rightHoles) == 0)
                                closestHole = closest(object, leftHoles)
                                if isFree(move(closestHole, Position(0, -1)), move(object.origin, Position(-1, 0)))
                                  new_object.origin = move(object.origin, unitVector(object, move(closestHole, Position(0, -1))))                      
                                end
                              else
                                closestLeftHole = closest(object, leftHoles)
                                closestRightHole = closest(object, rightHoles)
                                if distance(object.origin, closestLeftHole) > distance(object.origin, closestRightHole)
                                  if isFree(move(object.origin, Position(1, 0)), move(closestRightHole, Position(0, -1)))
                                    new_object.origin = move(object.origin, unitVector(new_object, move(closestRightHole, Position(0, -1))))
                                  elseif isFree(move(closestLeftHole, Position(0, -1)), move(object.origin, Position(-1, 0)))
                                    new_object.origin = move(object.origin, unitVector(new_object, move(closestLeftHole, Position(0, -1))))
                                  end
                                else
                                  if isFree(move(closestLeftHole, Position(0, -1)), move(object.origin, Position(-1, 0)))
                                    new_object.origin = move(object.origin, unitVector(new_object, move(closestLeftHole, Position(0, -1))))
                                  elseif isFree(move(object.origin, Position(1, 0)), move(closestRightHole, Position(0, -1)))
                                    new_object.origin = move(object.origin, unitVector(new_object, move(closestRightHole, Position(0, -1))))
                                  end
                                end
                              end
                            end
                          end
                          new_object
                        end

                        function nextSolid(object::Object)::Object 
                          # # println("nextSolid")
                          new_object = deepcopy(object)
                          if (isWithinBounds(move(object, Position(0, 1))) && reduce(&, map(x -> isFree(x, object), map(cell -> move(cell.position, Position(0, 1)), render(object)))))
                            new_object.origin = move(object.origin, Position(0, 1))
                          end
                          new_object
                        end
                        
                        function closest(object::Object, positions::Array{Position})::Position
                          closestDistance = sort(map(pos -> distance(pos, object.origin), positions))[1]
                          closest = filter(pos -> distance(pos, object.origin) == closestDistance, positions)[1]
                          closest
                        end

                        function isFree(start::Position, stop::Position)::Bool 
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                          else
                            GRID_SIZE_X = GRID_SIZE
                            GRID_SIZE_Y = GRID_SIZE
                          end
                          translated_start = GRID_SIZE_X * start.y + start.x 
                          translated_stop = GRID_SIZE_X * stop.y + stop.x
                          if translated_start < translated_stop
                            ordered_start = translated_start
                            ordered_end = translated_stop
                          else
                            ordered_start = translated_stop
                            ordered_end = translated_start
                          end
                          nums = [ordered_start:ordered_end;]
                          reduce(&, map(num -> isFree(Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X))), nums))
                        end

                        function isFree(start::Position, stop::Position, object::Object)::Bool 
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                          else
                            GRID_SIZE_X = GRID_SIZE
                            GRID_SIZE_Y = GRID_SIZE
                          end
                          translated_start = GRID_SIZE_X * start.y + start.x 
                          translated_stop = GRID_SIZE_X * stop.y + stop.x
                          if translated_start < translated_stop
                            ordered_start = translated_start
                            ordered_end = translated_stop
                          else
                            ordered_start = translated_stop
                            ordered_end = translated_start
                          end
                          nums = [ordered_start:ordered_end;]
                          reduce(&, map(num -> isFree(Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), object), nums))
                        end

                        function isFree(position::Position, object::Object)
                          length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, 
                          render(Scene(filter(obj -> obj.id != object.id , state.scene.objects), state.scene.background)))) == 0
                        end

                        function isFree(object::Object, orig_object::Object)::Bool
                          reduce(&, map(x -> isFree(x, orig_object), map(cell -> cell.position, render(object))))
                        end

                        function allPositions()
                          GRID_SIZE = state.GRID_SIZEHistory[0]
                          if GRID_SIZE isa AbstractArray 
                            GRID_SIZE_X = GRID_SIZE[1]
                            GRID_SIZE_Y = GRID_SIZE[2]
                          else
                            GRID_SIZE_X = GRID_SIZE
                            GRID_SIZE_Y = GRID_SIZE
                          end
                          nums = [1:GRID_SIZE_X*GRID_SIZE_Y - 1;]
                          map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)
                        end

                        function unfold(A)
                          V = []
                          for x in A
                              for elt in x
                                push!(V, elt)
                              end
                          end
                          V
                        end
                      
                    end
])

# compile.jl
 
"compile `aexpr` into Expr"
function compiletojulia(aexpr::AExpr)::Expr

  # dictionary containing types/definitions of global variables, for use in constructing init func.,
  # next func., etcetera; the three categories of global variable are external, initnext, and lifted  
  historydata = Dict([("external" => [au"""(external (: click Click))""".args[1], au"""(external (: left KeyPress))""".args[1], au"""(external (: right KeyPress))""".args[1], au"""(external (: up KeyPress))""".args[1], au"""(external (: down KeyPress))""".args[1]]), # :typedecl aexprs for all external variables
               ("initnext" => []), # :assign aexprs for all initnext variables
               ("lifted" => []), # :assign aexprs for all lifted variables
               ("types" => Dict{Symbol, Any}([:click => :Click, :left => :KeyPress, :right => :KeyPress, :up => :KeyPress, :down => :KeyPress, :background => :String])), # map of global variable names (symbols) to types
               ("on" => []),
               ("objects" => [])]) 
               
  if (aexpr.head == :program)
    # handle AExpression lines
    lines = filter(x -> x !== :(), map(arg -> compile(arg, historydata, aexpr), aexpr.args))
    
    # construct STATE struct and initialize state::STATE
    statestruct = compilestatestruct(historydata)
    initstatestruct = compileinitstate(historydata)
    
    # handle init, next, prev, and built-in functions
    initnextfunctions = compileinitnext(historydata)
    prevfunctions = compileprevfuncs(historydata)
    builtinfunctions = compilebuiltin()

    # remove empty lines
    lines = filter(x -> x != :(), 
            vcat(builtinfunctions, lines, statestruct, initstatestruct, prevfunctions, initnextfunctions))

    # construct module
    expr = quote
      module CompiledProgram
        export init, next
        import Base.min
        using Distributions
        using MLStyle: @match
        using Random
        rng = Random.GLOBAL_RNG
        $(lines...)
      end
    end  
    expr.head = :toplevel
    striplines(expr) 
  else
    throw(AutumnError("AExpr Head != :program"))
  end
end

"Run `prog` for finite number of time steps"
function runprogram(prog::Expr, n::Int)
  mod = eval(prog)
  state = mod.init(mod.Click(5, 5))

  for i in 1:n
    externals = [nothing, mod.Click(rand([1:10;]), rand([1:10;]))]
    state = mod.next(mod.next(state, externals[rand(Categorical([0.7, 0.3]))]))
  end
  state
end

# dynamics.jl

env = Dict(["custom_types" => Dict([
            "Object1" => [],
            "Object2" => [],
            "Object3" => [],              
            ]),
            "variables" => Dict([
              "object1" => "Object_Object2",
              "object2" => "Object_Object3",
              "objectlist1" => "ObjectList_Object3",
            ])])

function genUpdateRule(var, environment; p=0.7)
  if environment["variables"][var] == "Int"
    genInt(environment)
  elseif environment["variables"][var] == "Bool"
    genBoolUpdateRule(var, environment)
  elseif occursin("Object_", environment["variables"][var])
    genObjectUpdateRule(var, environment, p=p)
  else
    genObjectListUpdateRule(var, environment, p=p)
  end
end

# -----begin object generator + helper functions ----- #
function genObject(environment; p=0.9)
  object = genObjectName(environment)
  genObjectUpdateRule(object, environment, p=p)
end

function genObjectName(environment)
  objects = filter(var -> occursin("Object_", environment["variables"][var]), collect(keys(environment["variables"])))
  objects[rand(1:length(objects))]
end

function genObjectUpdateRule(object, environment; p=0.3)
  prob = rand()
  if prob < p
    if object == "obj"
      "$(object)"
    else
      "(prev $(object))"
    end
  else
    choices = [
      ("moveLeftNoCollision", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveRightNoCollision", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveUpNoCollision", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveDownNoCollision", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      # ("moveNoCollision", [:(genObjectUpdateRule($(object), $(environment))), :(genPosition($(environment)))]),
      # ("nextLiquid", [:(genObjectUpdateRule($(object), $(environment)))]),
      # ("nextSolid", [:(genObjectUpdateRule($(object), $(environment)))]),
      # ("rotate", [:(genObjectUpdateRule($(object), $(environment)))]),
      # ("rotateNoCollision", [:(genObjectUpdateRule($(object), $(environment)))]),
    ]
    choice = choices[rand(1:length(choices))]
    "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
  end
end

# ----- end object generator + helper functions ----- #

# ----- Int generator ----- # 
function genInt(environment)
  int_vars = map(v -> "(prev $(v))", filter(var -> environment["variables"][var] == "Int", collect(keys(environment["variables"]))))
  choices = [ #= fieldsFromCustomTypes("Int", environment)..., =# collect(1:5)..., int_vars...]
  choice = rand(choices)
  if (choice isa String) || (choice isa Int)
    choice
  else
    @show "($(choice[1]) $(join(map(eval, choice[2]), " ")))"  
    "($(choice[1]) $(join(map(eval, choice[2]), " ")))"      
  end
end

# ----- Bool generator ----- #
function genBool(environment)
  choices = [
    ("clicked", []),
    ("clicked", [:(genPosition($(environment)))]),
    ("left", []),
    ("right", []),
    ("up", []),
    ("down", []),
    ("true", []), # TODO: add not, or, and -- need to be able to specify prior probabilities 
  ]
  if length(filter(var -> occursin("Object_", environment["variables"][var]), collect(keys(environment["variables"])))) > 0
    push!(choices, [("clicked", [:(genObject($(environment), p=1.0))]),
                    ("intersects", [:(genObject($(environment), p=1.0)), :(genObject($(environment), p=1.0))]),
                    ("isWithinBounds", [:(genObject($(environment)))])
                   ]...)
  end

  bool_vars = map(v -> "(prev $(v))", filter(var -> environment["variables"][var] == "Bool", collect(keys(environment["variables"]))))
  foreach(var -> push!(choices, (var, [])), bool_vars)

  push!(choices, fieldsFromCustomTypes("Bool", environment)...)

  choice = choices[rand(1:length(choices))]
  if (length(choice[2]) == 0)
    "$(choice[1])"
  else
    "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
  end 

end

function genBoolUpdateRule(bool, environment)
  bool_val = "(prev $(bool))"
  rand(["(! $(bool_val))", bool_val])
end

# ----- Position generator ----- #
function genPosition(environment)
  choices = [
    ("Position", [:(genInt($(environment))), :(genInt($(environment)))]),
    ("displacement", [:(genPosition($(environment))), :(genPosition($(environment)))]),
    ("unitVector", [:(genPosition($(environment)))]),
    ("uniformChoice", [:(genPositionList($(environment)))])
  ]
  # if object constants exist, add support for (.. obj origin)
  if length(filter(var -> occursin("Object_", environment["variables"][var]), collect(keys(environment["variables"])))) > 0
    push!(choices, ("..", [:(genObject($(environment), p=1.0)), :(String("origin"))]))    
  end

  choice = choices[rand(1:length(choices))]
  "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
end

# ----- Click generator ----- #
function genClick(environment)
  options = [
    "click"
  ]
  choice = choices[rand(1:length(choices))]
  "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
end

# ----- Position List generator ----- #
function genPositionList(environment)
  choices = [
    ("randomPositions", ["GRID_SIZE", :(genInt($(environment)))])
  ]
  if length(filter(var -> occursin("Object_", environment["variables"][var]), collect(keys(environment["variables"])))) > 0
    push!(choices, ("adjPositions", [:(genObject($(environment)))]))
  end
  choice = choices[rand(1:length(choices))]
  "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
end

# ----- begin object list generator + helper functions ----- #
function genObjectList(environment)
  object_list = genObjectListName(environment)
  genObjectListUpdateRule(object_list, environment)
end

function genObjectListName(environment)
  object_lists = filter(var -> occursin("ObjectList_", environment["variables"][var]), collect(keys(environment["variables"])))
  object_lists[rand(1:length(object_lists))]
end

function genObjectConstructor(type, environment)
  new_type = occursin("_", type) ? type : string("Object_", type)
  constructor = map(tuple -> Meta.parse("gen$(tuple[2])($(environment))"), environment["custom_types"][new_type])
  push!(constructor, :(genPosition($(environment))))
  "($(type) $(join(map(eval, constructor), " ")))"
end

function genObject(type, environment, p=0.9)
  objects_with_type = filter(var -> environment["variables"][var] == type, collect(keys(environment["variables"])))
  prob = rand()
  if (prob < p) && length(objects_with_type) != 0
    rand(objects_with_type)
  else
    constructor = genObjectConstructor(type, environment)
    constructor 
  end
end

function genObjectListUpdateRule(object_list, environment; p=0.7)
  prob = rand()
  if prob < p
    "(prev $(object_list))"
  else
    choices = [
      ("addObj", 
        [:(genObjectListUpdateRule($(object_list), $(environment))), 
         :(genObjectConstructor($(String(split(environment["variables"][object_list], "_")[end])), $(environment))),
        ]
      ),
      ("updateObj", 
        [:(genObjectListUpdateRule($(object_list), $(environment))),
         :(genLambda($(environment)))
        ]),
    ]
    choice = choices[rand(1:length(choices))]
    "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
  end
end

function genLambda(environment)
  choice = ("-->", [:(String("obj")), :(genObjectUpdateRule("obj", $(environment)))])
  "($(choice[1]) $(join(map(eval, choice[2]), " ")))"
end
# ----- end object list generator + helper functions ----- #

# ----- string generator ----- #
function genString(environment)
  colors = ["red", "yellow", "green", "blue"]
  color = colors[rand(1:length(colors))]
  """ "$(color)" """
end

# ----- helper functions ----- #

function fieldsFromCustomTypes(fieldtype::String, environment)
  branches = []
  types_with_field = filter(type -> fieldtype in map(tuple -> tuple[2], environment["custom_types"][type]), collect(keys(environment["custom_types"])))
  for type in types_with_field
    fieldnames = map(t -> t[1], filter(tuple -> tuple[2] == fieldtype, environment["custom_types"][type]))
    if length(filter(var -> environment["variables"][var] == type, collect(keys(environment["variables"])))) > 0  
      foreach(fieldname -> push!(branches, ("..", [Meta.parse("genObject(\"$(split(type, "_")[end])\", $(environment))"), :(String($(fieldname)))])), fieldnames)
    end
  end
  branches
end

# generativemodel.jl

"""Generate program"""
function generateprogram(rng=Random.GLOBAL_RNG; gridsize::Int=16, group::Bool=false)
  # generate objects and types 
  types_and_objects = generatescene_objects(rng, gridsize=gridsize)
  generateprogram_given_objects(types_and_objects, rng, gridsize=gridsize, group=group)
end

"""Generate program given object decomposition (types and objects)"""
function generateprogram_given_objects(types_and_objects, rng=Random.GLOBAL_RNG; gridsize::Int=16, group::Bool=false)
  # generate objects and types 
  types, objects, background, _ = types_and_objects

  non_object_global_vars = []
  num_non_object_global_vars = rand(0:0)

  for i in 1:num_non_object_global_vars
    type = rand(["Bool", "Int"])
    if type == "Bool"
      push!(non_object_global_vars, (type, rand(["true", "false"]), i))
    else
      push!(non_object_global_vars, (type, rand(1:3), i))
    end
  end

  if (!group)
    # construct environment object
    environment = Dict(["custom_types" => Dict(
                                              map(t -> "Object_ObjType$(t.id)" => t.custom_fields, types) 
                                              ),
                        "variables" => Dict(
                                            vcat(
                                            map(obj -> "obj$(obj.id)" => "Object_ObjType$(obj.type.id)", objects)...,                    
                                            map(tuple -> "globalVar$(tuple[3])" => tuple[1], non_object_global_vars)...
                                            )
                                           )])
    
    # generate next values for each object
    next_vals = map(obj -> genObjectUpdateRule("obj$(obj.id)", environment), objects)
    objects = [(objects[i], next_vals[i]) for i in 1:length(objects)]

    # generate on-clauses for each object
    on_clause_object_ids = rand(1:length(objects), rand(1:length(objects)))
    on_clauses = map(i -> (genBool(environment), genUpdateRule("obj$(i)", environment, p=0.5), i), on_clause_object_ids)

    # generate on-clauses for each non-object global variable
    # generate next values for each non-object global variable
    if length(non_object_global_vars) != 0
      non_object_nexts = map(tuple -> genUpdateRule("globalVar$(tuple[3])", environment), non_object_global_vars)
      non_object_on_clause_ids = rand(1:length(non_object_global_vars), rand(0:length(non_object_global_vars)))
      non_object_on_clauses = map(i -> (genBool(environment), genUpdateRule("globalVar$(i)", environment), i), non_object_on_clause_ids)
    else
      non_object_nexts = []
      non_object_on_clauses = []
    end

    """
    (program
      (= GRID_SIZE $(gridsize))
      (= background "$(background)")
      $(join(map(t -> "(object ObjType$(t.id) $(join(map(field -> "(: $(field[1]) $(field[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) "$(t.color)")""", t.shape), " "))))", types), "\n  "))

      $(join(map(tuple -> "(: globalVar$(tuple[3]) $(tuple[1]))", non_object_global_vars), "\n  "))

      $(join(map(tuple -> "(= globalVar$(tuple[3]) (initnext $(tuple[2]) $(non_object_nexts[tuple[3]])))", non_object_global_vars), "\n  "))

      $((join(map(obj -> """(: obj$(obj[1].id) ObjType$(obj[1].type.id))""", objects), "\n  "))...)

      $((join(map(obj -> 
      """(= obj$(obj[1].id) (initnext (ObjType$(obj[1].type.id) $(join(obj[1].custom_field_values, " ")) (Position $(obj[1].position[1] - 1) $(obj[1].position[2] - 1))) $(obj[2])))""", objects), "\n  ")))

      $((join(map(tuple -> 
      """(on $(tuple[1]) (= obj$(tuple[3]) $(tuple[2])))""", on_clauses), "\n  "))...)

      $((join(map(tuple -> 
      """(on $(tuple[1]) (= globalVar$(tuple[3]) $(tuple[2])))""", non_object_on_clauses), "\n  "))...)
    )
    """
  else
    # group objects of the same type into lists
    type_ids = unique(map(obj -> obj.type.id, objects))
    list_type_ids = filter(id -> count(obj -> obj.type.id == id, objects) > 1, type_ids)
    constant_type_ids = filter(id -> count(obj -> obj.type.id == id, objects) == 1, type_ids)

    println(length(types))
    println(length(objects))

    environment = Dict(["custom_types" => Dict(
                                map(t -> "Object_ObjType$(t.id)" => t.custom_fields, types) 
                                ),
                        "variables" => Dict(
                              vcat(
                                map(id -> "objList$(findall(x -> x == id, list_type_ids)[1])" => "ObjectList_ObjType$(id)", list_type_ids)...,
                                map(id -> "obj$(findall(x -> x == id, constant_type_ids)[1])" => "Object_ObjType$(id)", constant_type_ids)...,
                                map(tuple -> "globalVar$(tuple[3])" => tuple[1], non_object_global_vars)...       
                              )             
                            )])

    # generate next values and on-clauses for each object
    # lists
    if length(list_type_ids) != 0
      next_list_vals = map(id -> genUpdateRule("objList$(findall(x -> x == id, list_type_ids)[1])", environment), list_type_ids)

      on_clause_list_ids = rand(list_type_ids, rand(1:length(list_type_ids)))
      on_clauses_list = map(id -> (genBool(environment), genUpdateRule("objList$(findall(x -> x == id, list_type_ids)[1])", environment, p=0.5), findall(x -> x == id, list_type_ids)[1]), on_clause_list_ids)
    else
      next_list_vals = []
      on_clauses_list = []
    end

    # constants
    if length(constant_type_ids) != 0
      next_constant_vals = map(id -> genUpdateRule("obj$(findall(x -> x == id, constant_type_ids)[1])", environment), constant_type_ids)
      
      on_clauses_constant_ids = rand(constant_type_ids, rand(1:length(constant_type_ids)))
      on_clauses_constant = map(id -> (genBool(environment), genUpdateRule("obj$(findall(x -> x == id, constant_type_ids)[1])", environment, p=0.5), findall(x -> x == id, constant_type_ids)[1]), on_clauses_constant_ids)
    else
      next_constant_vals = []
      on_clauses_constant = []
    end

    # generate next values and on-clauses for each non-object variable
    if length(non_object_global_vars) != 0
      non_object_nexts = map(tuple -> genUpdateRule("globalVar$(tuple[3])", environment), non_object_global_vars)
      non_object_on_clause_ids = rand(1:length(non_object_global_vars), rand(0:length(non_object_global_vars)))
      non_object_on_clauses = map(i -> (genBool(environment), genUpdateRule("globalVar$(i)", environment), i), non_object_on_clause_ids)
    else
      non_object_nexts = []
      non_object_on_clauses = []
    end
    """
    (program
      (= GRID_SIZE $(gridsize))
      (= background "$(background)")
      $(join(map(t -> "(object ObjType$(t.id) $(join(map(field -> "(: $(field[1]) $(field[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) "$(t.color)")""", t.shape), " "))))", types), "\n  "))

      $(join(map(tuple -> "(: globalVar$(tuple[3]) $(tuple[1]))", non_object_global_vars), "\n  "))

      $(join(map(tuple -> "(= globalVar$(tuple[3]) (initnext $(tuple[2]) $(non_object_nexts[tuple[3]])))", non_object_global_vars), "\n  "))

      $((join(map(id -> """(: objList$(findall(x -> x == id, list_type_ids)[1]) (List ObjType$(id)))""", list_type_ids), "\n  "))...)
      $((join(map(id -> """(: obj$(findall(x -> x == id, constant_type_ids)[1]) ObjType$(id))""", constant_type_ids), "\n  "))...)

      $((join(map(id -> 
      """(= objList$(findall(x -> x == id, list_type_ids)[1]) (initnext (list $(join(map(obj -> "(ObjType$(obj.type.id) $(join(obj.custom_field_values, " ")) (Position $(obj.position[1] - 1) $(obj.position[2] - 1)))", filter(o -> o.type.id == id, objects)), " "))) $(next_list_vals[findall(y -> y == id, list_type_ids)[1]])))""", list_type_ids), "\n  ")))

      $((join(map(id -> 
      """(= obj$(findall(x -> x == id, constant_type_ids)[1]) (initnext $(join(map(obj -> "(ObjType$(obj.type.id) $(join(obj.custom_field_values, " ")) (Position $(obj.position[1] - 1) $(obj.position[2] - 1)))", filter(o -> o.type.id == id, objects)))) $(next_constant_vals[findall(y -> y == id, constant_type_ids)[1]])))""", constant_type_ids), "\n  ")))

      $((join(map(tuple -> 
      """(on $(tuple[1]) (= objList$(tuple[3]) $(tuple[2])))""", on_clauses_list), "\n  "))...)

      $((join(map(tuple -> 
      """(on $(tuple[1]) (= obj$(tuple[3]) $(tuple[2])))""", on_clauses_constant), "\n  "))...)

      $((join(map(tuple -> 
      """(on $(tuple[1]) (= globalVar$(tuple[3]) $(tuple[2])))""", non_object_on_clauses), "\n  "))...)
    )
    """
  end
end

function generate_hypothesis_update_rule(object, object_decomposition; p=0.3)
  types, objects, background, gridsize = object_decomposition
  objects = [object, objects...]
  # construct environment 
  environment = Dict(["custom_types" => Dict(map(t -> "Object_ObjType$(t.id)" => t.custom_fields, types) 
                          ),
                      "variables" => Dict(map(obj -> "obj$(obj.id)" => "Object_ObjType$(obj.type.id)", objects)
                      )])
 
  # generate update rule 
  """(= obj$(object.id) $(genObjectUpdateRule("obj$(object.id)", environment, p=p)))"""
end

# singletimestepsolution.jl

"""Construct matrix of single timestep solutions"""
function singletimestepsolution_matrix(observations, grid_size)
  object_decomposition = parse_and_map_objects(observations)
  object_types, object_mapping, background, _ = object_decomposition
  # matrix of update function sets for each object/time pair
  # number of rows = number of objects, number of cols = number of time steps  
  num_objects = length(collect(keys(object_mapping)))
  matrix = [[] for object_id in 1:num_objects, time in 1:(length(observations) - 1)]
  prev_used_rules = []
  @show size(matrix)
  # for each subsequent frame, map objects  
  for time in 2:length(observations)
    # for each object in previous time step, determine a set of update functions  
    # that takes the previous object to the next object
    for object_id in 1:num_objects
      update_functions, prev_used_rules = synthesize_update_functions(object_id, time, object_decomposition, prev_used_rules, grid_size)
      @show update_functions 
      matrix[object_id, time - 1] = update_functions
    end
  end
  matrix, object_decomposition, prev_used_rules
end

expr = nothing
mod = nothing
global_iters = 0
"""Synthesize a set of update functions that """
function synthesize_update_functions(object_id, time, object_decomposition, prev_used_rules, grid_size=16, max_iters=50)
  object_types, object_mapping, background, _ = object_decomposition
  @show object_id 
  @show time
  prev_object = object_mapping[object_id][time - 1]
  
  next_object = object_mapping[object_id][time]
  #@show object_id 
  #@show time
  #@show prev_object 
  #@show next_object
  # @show isnothing(prev_object) && isnothing(next_object)
  if isnothing(prev_object) && isnothing(next_object)
    [""], prev_used_rules
  elseif isnothing(prev_object)
    ["(= addedObjType$(next_object.type.id)List (addObj addedObjType$(next_object.type.id)List (ObjType$(next_object.type.id) (Position $(next_object.position[1]) $(next_object.position[2])))))"], prev_used_rules
  elseif isnothing(next_object)
    if object_mapping[object_id][1] == nothing # object was added later; contained in addedList
      ["(= addedObjType$(prev_object.type.id)List (removeObj addedObjType$(prev_object.type.id)List (--> obj (== (.. obj id) $(object_id)))))"], prev_used_rules
    else # object was present at the start of the program
      ["(= obj$(object_id) (removeObj obj$(object_id)))"], prev_used_rules
    end
  else # actual synthesis problem
    prev_objects = filter(obj -> !isnothing(obj) && (obj.id != prev_object.id), [object_mapping[id][time - 1] for id in 1:length(collect(keys(object_mapping)))])
    #@show prev_objects
    solutions = []
    iters = 0
    prev_used_rules_index = 1
    using_prev = false
    while length(solutions) != 1 && iters < max_iters
      hypothesis_program = program_string_synth((object_types, sort([prev_objects..., prev_object], by=(x -> x.id)), background, grid_size))
      
      if prev_used_rules_index <= length(prev_used_rules)
        update_rule = replace(prev_used_rules[prev_used_rules_index], "objX" => "obj$(object_id)")
        using_prev = true
        prev_used_rules_index += 1
      else
        using_prev = false
        update_rule = generate_hypothesis_update_rule(prev_object, (object_types, prev_objects, background, grid_size), p=0.2) # "(= obj1 (moveDownNoCollision (moveDownNoCollision (prev obj1))))"
      end      

      hypothesis_program = string(hypothesis_program[1:end-2], "\n\t (on true ", update_rule, ")\n)")
      # println(hypothesis_program)
      # @show global_iters
      # @show update_rule

      global expr = striplines(compiletojulia(parseautumn(hypothesis_program)))
      @show expr
      module_name = Symbol("CompiledProgram$(global_iters)")
      global expr.args[1].args[2] = module_name
      # @show expr.args[1].args[2]
      global mod = @eval $(expr)
      # @show repr(mod)
      hypothesis_frame_state = @eval mod.next(mod.init(nothing, nothing, nothing, nothing, nothing), nothing, nothing, nothing, nothing, nothing)
      
      hypothesis_object = filter(o -> o.id == object_id, hypothesis_frame_state.scene.objects)[1]
      #@show hypothesis_frame_state.scene.objects
      #@show hypothesis_object

      if render_equals(hypothesis_object, next_object)
        if using_prev
          println("HOORAY")
        end
        generic_update_rule = replace(update_rule, "obj$(object_id)" => "objX")
        if !(generic_update_rule in prev_used_rules)
          push!(prev_used_rules, generic_update_rule)
        end

        if object_mapping[object_id][1] == nothing # object was added later; contained in addedList
          map_lambda_func = replace(string("(-->", replace(update_rule, "obj$(object_id)" => "obj")[3:end]), "(prev obj)" => "obj")
          push!(solutions, "(= addedObjType$(prev_object.type.id)List (updateObj addedObjType$(prev_object.type.id)List $(map_lambda_func) (--> obj (== (.. obj id) $(object_id)))))")
        else # object was present at the start of the program
          push!(solutions, update_rule)
        end
      end
      
      iters += 1
      global global_iters += 1
      
    end
    if (iters == max_iters)
      println("FAILURE")
    end
    solutions, prev_used_rules
  end
end

"""Parse observations into object types and objects, and assign 
   objects in current observed frame to objects in next frame"""
function parse_and_map_objects(observations)
  object_mapping = Dict{Int, Array{Union{Nothing, Obj}}}()

  # initialize object mapping with object_decomposition from first observation
  object_types, objects, background, dim = parsescene_autumn(observations[1]) # parsescene_autumn_singlecell
  for object in objects
    object_mapping[object.id] = [object]
  end

  for time in 2:length(observations)
    next_object_types, next_objects, _, _ = parsescene_autumn(observations[time]) # parsescene_autumn_singlecell

    # update object_types with new elements in next_object_types 
    new_object_types = filter(type -> !((repr(sort(type.shape)), type.color) in map(t -> (repr(sort(t.shape)), t.color), object_types)), next_object_types)
    @show object_types 
    @show new_object_types
    if length(new_object_types) != 0
      for i in 1:length(new_object_types)
        new_type = new_object_types[i]
        new_type.id = length(object_types) + i
        push!(object_types, new_type)
      end
    end

    # reassign type ids in next_objects according to global type set (object_types)
    for object in next_objects
      object.type = filter(type -> (type.shape, type.color) == (object.type.shape, object.type.color), object_types)[1]
    end

    # construct mapping between objects and next_objects
    for type in object_types
      curr_objects_with_type = filter(o -> o.type.id == type.id, objects)
      next_objects_with_type = filter(o -> o.type.id == type.id, next_objects)
      
      closest_objects = compute_closest_objects(curr_objects_with_type, next_objects_with_type)
      while !(isempty(curr_objects_with_type) || isempty(next_objects_with_type)) 
        for (object_id, closest_ids) in closest_objects
          if length(intersect(closest_ids, map(o -> o.id, next_objects_with_type))) == 1
            closest_id = intersect(closest_ids, map(o -> o.id, next_objects_with_type))[1]
            next_object = filter(o -> o.id == closest_id, next_objects_with_type)[1]

            # remove curr and next objects from respective lists
            filter!(o -> o.id != object_id, curr_objects_with_type)
            filter!(o -> o.id != closest_id, next_objects_with_type)
            delete!(closest_objects, object_id)
            
            # add next object to mapping
            next_object.id = object_id
            push!(object_mapping[object_id], next_object)
          end

          if length(collect(keys(filter(pair -> length(intersect(last(pair), map(o -> o.id, next_objects_with_type))) == 1, closest_objects)))) == 0
            # every remaining object to be mapped is equidistant to at least two closest objects, or zero objects
            # perform a brute force assignment
            while !isempty(curr_objects_with_type) && !isempty(next_objects_with_type)
              # do something
              object = curr_objects_with_type[1]
              next_object = next_objects_with_type[1]
              @show curr_objects_with_type
              @show next_objects_with_type
              curr_objects_with_type = filter(o -> o.id != object.id, curr_objects_with_type)
              next_objects_with_type = filter(o -> o.id != next_object.id, next_objects_with_type)
              
              next_object.id = object.id
              push!(object_mapping[object.id], next_object)
            end
            break
          end
        end
      end

      max_id = length(collect(keys(object_mapping)))
      if isempty(curr_objects_with_type) && !(isempty(next_objects_with_type))
        # handle addition of objects
        for i in 1:length(next_objects_with_type)
          next_object = next_objects_with_type[i]
          next_object.id = max_id + i
          object_mapping[next_object.id] = [[nothing for i in 1:(time - 1)]..., next_object]
        end
      elseif !(isempty(curr_objects_with_type)) && isempty(next_objects_with_type)
        # handle removal of objects
        for object in curr_objects_with_type
          push!(object_mapping[object.id], [nothing for i in time:length(observations)]...)
        end
      end
    end

    objects = next_objects

  end
  (object_types, object_mapping, background, dim)  
end

function compute_closest_objects(curr_objects, next_objects)
  closest_objects = Dict{Int, AbstractArray}()
  for object in curr_objects
    distances = map(o -> distance(object.position, o.position), next_objects)
    closest_objects[object.id] = map(obj -> obj.id, filter(o -> distance(object.position, o.position) == minimum(distances), next_objects))
  end
  closest_objects
end

function distance(pos1, pos2)
  pos1_x, pos1_y = pos1
  pos2_x, pos2_y = pos2
  # sqrt(Float((pos1_x - pos2_x)^2 + (pos1_y - pos2_y)^2))
  abs(pos1_x - pos2_x) + abs(pos1_y - pos2_y)
end

function render_equals(hypothesis_object, actual_object)
  translated_hypothesis_object = map(cell -> (cell.position.x + hypothesis_object.origin.x, cell.position.y + hypothesis_object.origin.y), hypothesis_object.render)
  translated_actual_object = map(pos -> (pos[1] + actual_object.position[1], pos[2] + actual_object.position[2]), actual_object.type.shape)
  translated_hypothesis_object == translated_actual_object
end

function generate_observations(m::Module)
  state = m.init(nothing, nothing, nothing, nothing, nothing)
  observations = []
  for i in 0:10
    if i % 3 == 2
      # state = m.next(state, nothing, nothing, nothing, nothing, nothing)
      state = m.next(state, m.Click(rand(5:10), rand(5:10)), nothing, nothing, nothing, nothing)
    else
      state = m.next(state, nothing, nothing, nothing, nothing, nothing)
    end
    push!(observations, m.render(state.scene))
  end
  observations
end

function singletimestepsolution_program(observations, grid_size=16)
  
  matrix, object_decomposition, _ = singletimestepsolution_matrix(observations, grid_size)
  singletimestepsolution_program_given_matrix(matrix, object_decomposition, grid_size)
end

function singletimestepsolution_program_given_matrix(matrix, object_decomposition, grid_size=16)
  object_types, object_mapping, background, _ = object_decomposition
  
  objects = sort(filter(obj -> obj != nothing, [object_mapping[i][1] for i in 1:length(collect(keys(object_mapping)))]), by=(x -> x.id))
  
  program_no_update_rules = program_string_synth((object_types, objects, background, grid_size))
  
  list_variables = join(map(type -> 
  """(: addedObjType$(type.id)List (List ObjType$(type.id)))\n  (= addedObjType$(type.id)List (initnext (list) (prev addedObjType$(type.id)List)))\n""", 
  object_types),"\n  ")
  
  time = """(: time Int)\n  (= time (initnext 0 (+ time 1)))"""

  update_rule_times = filter(time -> join(map(l -> l[1], matrix[:, time]), "") != "", [1:size(matrix)[2]...])
  update_rules = join(map(time -> """(on (== time $(time))\n  (let\n    ($(join(map(l -> l[1], matrix[:, time]), "\n    "))))\n  )""", update_rule_times), "\n  ")
  
  string(program_no_update_rules[1:end-2], 
        "\n\n  $(list_variables)",
        "\n\n  $(time)", 
        "\n\n  $(update_rules)", 
        ")")
end

# autumnstdlib.jl 

update_nt(@nospecialize(Γ::NamedTuple), x::Symbol, v) = merge(Γ, NamedTuple{(x,)}((v,)))

abstract type Object end
abstract type KeyPress end

struct Left <: KeyPress end
struct Right <: KeyPress end
struct Up <: KeyPress end
struct Down <: KeyPress end

struct Click
  x::Int
  y::Int                    
end

Click(x, y, @nospecialize(state::NamedTuple)) = Click(x, y)

struct Position
  x::Int
  y::Int
end

Position(x, y, @nospecialize(state::NamedTuple)) = Position(x, y) 

struct Cell 
  position::Position
  color::String
  opacity::Float64
end

Cell(position::Position, color::String) = Cell(position, color, 0.8)
Cell(x::Int, y::Int, color::String) = Cell(Position(floor(Int, x), floor(Int, y)), color, 0.8)
# Cell(x::Int, y::Int, color::String, opacity::Float64) = Cell(Position(floor(Int, x), floor(Int, y)), color, opacity)

Cell(x, y, color::String, @nospecialize(state::NamedTuple)) = Cell(floor(Int, x), floor(Int, y), color)
Cell(position::Position, color::String, @nospecialize(state::NamedTuple)) = Cell(position::Position, color::String)

# struct Scene
#   objects::Array{Object}
#   background::String
# end

# Scene(@nospecialize(objects::AbstractArray)) = Scene(objects, "#ffffff00")

# function render(scene)::Array{Cell}
#   vcat(map(obj -> render(obj), filter(obj -> obj.alive, scene.objects))...)
# end

function prev(@nospecialize(obj::NamedTuple), @nospecialize(state))
  prev_objects = filter(o -> o.id == obj.id, state.scene.objects)
  if prev_objects != []
    prev_objects[1]                            
  else
    obj
  end
end

function render(@nospecialize(obj::NamedTuple), @nospecialize(state=nothing))::Array{Cell}
  if !(:id in keys(obj))
    vcat(map(o -> render(o), filter(x -> x.alive, obj.objects))...)
  else
    if obj.alive
      map(cell -> Cell(move(cell.position, obj.origin), cell.color), obj.render)
    else
      []
    end
  end
end


function occurred(click, @nospecialize(state=nothing))
  !isnothing(click)
end

function uniformChoice(freePositions, @nospecialize(state::NamedTuple))
  freePositions[rand(state.rng, Categorical(ones(length(freePositions))/length(freePositions)))]
end

function uniformChoice(freePositions, n::Union{Int, BigInt}, @nospecialize(state::NamedTuple))
  map(idx -> freePositions[idx], rand(state.rng, Categorical(ones(length(freePositions))/length(freePositions)), n))
end

function min(arr, @nospecialize(state=nothing))
  Base.min(arr...)
end

function range(start::Int, stop::Int, @nospecialize(state=nothing))
  [start:stop;]
end

function isWithinBounds(@nospecialize(obj::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  # # println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
  length(filter(cell -> !isWithinBounds(cell.position, state), render(obj))) == 0
end

function clicked(click::Union{Click, Nothing}, @nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  if isnothing(click)
    false
  else
    GRID_SIZE = state.GRID_SIZEHistory[0]
    if GRID_SIZE isa AbstractArray 
      GRID_SIZE_X = GRID_SIZE[1]
      nums = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(object))
      (GRID_SIZE_X * click.y + click.x) in nums
    else
      nums = map(cell -> GRID_SIZE*cell.position.y + cell.position.x, render(object))
      (GRID_SIZE * click.y + click.x) in nums
    end

  end
end

function clicked(click::Union{Click, Nothing}, @nospecialize(objects::AbstractArray), @nospecialize(state::NamedTuple))  
  # # println("LOOK AT ME")
  # # println(reduce(&, map(obj -> clicked(click, obj), objects)))
  if isnothing(click)
    false
  else
    foldl(|, map(obj -> clicked(click, obj, state), objects), init=false)
  end
end

function objClicked(click::Union{Click, Nothing}, @nospecialize(objects::AbstractArray), @nospecialize(state=nothing))::Union{NamedTuple, Nothing}
  # # println(click)
  if isnothing(click)
    nothing
  else
    clicked_objects = filter(obj -> clicked(click, obj, state), objects)
    if length(clicked_objects) == 0
      nothing
    else
      clicked_objects[1]
    end
  end

end

function clicked(click::Union{Click, Nothing}, x::Int, y::Int, @nospecialize(state::NamedTuple))::Bool
  if click == nothing
    false
  else
    click.x == x && click.y == y                         
  end
end

function clicked(click::Union{Click, Nothing}, pos::Position, @nospecialize(state::NamedTuple))::Bool
  if click == nothing
    false
  else
    click.x == pos.x && click.y == pos.y                         
  end
end

function intersects(@nospecialize(obj1::NamedTuple), @nospecialize(obj2::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj2))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj2))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj1::NamedTuple), @nospecialize(obj2::AbstractArray), @nospecialize(state::NamedTuple))::Bool
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj2::AbstractArray), @nospecialize(obj1::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  GRID_SIZE = state.GRID_SIZEHistory[0] 
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
    nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj1::AbstractArray), @nospecialize(obj2::AbstractArray), @nospecialize(state::NamedTuple))::Bool
  # # println("INTERSECTS")
  # # @show typeof(obj1) 
  # # @show typeof(obj2) 
  if (length(obj1) == 0) || (length(obj2) == 0)
    false  
  elseif (obj1[1] isa NamedTuple) && (obj2[1] isa NamedTuple)
    # # println("MADE IT")
    GRID_SIZE = state.GRID_SIZEHistory[0]
    if GRID_SIZE isa AbstractArray 
      GRID_SIZE_X = GRID_SIZE[1]
      nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj1)...))
      nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
      length(intersect(nums1, nums2)) != 0
    else
      nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj1)...))
      nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
      length(intersect(nums1, nums2)) != 0
    end
  else
    length(intersect(obj1, obj2)) != 0 
  end
end

function intersects(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  objects = state.scene.objects
  intersects(object, objects, state)
end

function addObj(@nospecialize(list::AbstractArray), @nospecialize(obj::NamedTuple), @nospecialize(state=nothing))
  obj = update_nt(obj, :changed, true)
  new_list = vcat(list, obj)
  new_list
end

function addObj(@nospecialize(list::AbstractArray), @nospecialize(objs::AbstractArray), @nospecialize(state=nothing))
  objs = map(obj -> update_nt(obj, :changed, true), objs)
  new_list = vcat(list, objs)
  new_list
end

function removeObj(@nospecialize(list::AbstractArray), @nospecialize(obj::NamedTuple), @nospecialize(state=nothing))
  new_list = deepcopy(list)
  for x in filter(o -> o.id == obj.id, new_list)
    index = findall(o -> o.id == x.id, new_list)[1]
    new_list[index] = update_nt(update_nt(x, :alive, false), :changed, true)
    #x.alive = false 
    #x.changed = true
  end
  new_list
end

function removeObj(@nospecialize(list::AbstractArray), fn, @nospecialize(state=nothing))
  new_list = deepcopy(list)
  for x in filter(obj -> fn(obj), new_list)
    index = findall(o -> o.id == x.id, new_list)[1]
    new_list[index] = update_nt(update_nt(x, :alive, false), :changed, true)
    #x.alive = false 
    #x.changed = true
  end
  new_list
end

function removeObj(@nospecialize(obj::NamedTuple), @nospecialize(state=nothing))
  new_obj = deepcopy(obj)
  new_obj = update_nt(update_nt(new_obj, :alive, false), :changed, true)
  # new_obj.alive = false
  # new_obj.changed = true
  # new_obj
end

function updateObj(@nospecialize(obj::NamedTuple), field::String, value, @nospecialize(state=nothing))
  fields = fieldnames(typeof(obj))
  custom_fields = fields[5:end-1]
  origin_field = (fields[2],)

  constructor_fields = (custom_fields..., origin_field...)
  constructor_values = map(x -> x == Symbol(field) ? value : getproperty(obj, x), constructor_fields)

  new_obj = typeof(obj)(constructor_values...)
  setproperty!(new_obj, :id, obj.id)
  setproperty!(new_obj, :alive, obj.alive)
  setproperty!(new_obj, :changed, true)

  setproperty!(new_obj, Symbol(field), value)
  state.objectsCreated -= 1    
  new_obj
end

function filter_fallback(@nospecialize(obj::NamedTuple), @nospecialize(state=nothing))
  true
end

# function updateObj(@nospecialize(list::AbstractArray), map_fn, filter_fn, @nospecialize(state=nothing))
#   orig_list = filter(obj -> !filter_fn(obj), list)
#   filtered_list = filter(filter_fn, list)
#   new_filtered_list = map(map_fn, filtered_list)
#   foreach(obj -> obj.changed = true, new_filtered_list)
#   vcat(orig_list, new_filtered_list)
# end

# function updateObj(@nospecialize(list::AbstractArray), map_fn, @nospecialize(state=nothing))
#   orig_list = filter(obj -> false, list)
#   filtered_list = filter(obj -> true, list)
#   new_filtered_list = map(map_fn, filtered_list)
#   foreach(obj -> obj.changed = true, new_filtered_list)
#   vcat(orig_list, new_filtered_list)
# end

function adjPositions(position::Position, @nospecialize(state::NamedTuple))::Array{Position}
  filter(x -> isWithinBounds(x, state), [Position(position.x, position.y + 1), Position(position.x, position.y - 1), Position(position.x + 1, position.y), Position(position.x - 1, position.y)])
end

function isWithinBounds(position::Position, @nospecialize(state::NamedTuple))::Bool
  GRID_SIZE = state.GRID_SIZEHistory[0] 
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  (position.x >= 0) && (position.x < GRID_SIZE_X) && (position.y >= 0) && (position.y < GRID_SIZE_Y)                  
end

function isFree(position::Position, @nospecialize(state::NamedTuple))::Bool
  length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, render(state.scene))) == 0
end

function isFree(click::Union{Click, Nothing}, @nospecialize(state::NamedTuple))::Bool
  if click == nothing
    false
  else
    isFree(Position(click.x, click.y), state)
  end
end

function rect(pos1::Position, pos2::Position, @nospecialize(state=nothing))
  min_x = pos1.x 
  max_x = pos2.x 
  min_y = pos1.y
  max_y = pos2.y 

  positions = []
  for x in min_x:max_x 
    for y in min_y:max_y
      push!(positions, Position(x, y))
    end
  end
  positions
end

function unitVector(position1::Position, position2::Position, @nospecialize(state::NamedTuple))::Position
  deltaX = position2.x - position1.x
  deltaY = position2.y - position1.y
  if (floor(Int, abs(sign(deltaX))) == 1 && floor(Int, abs(sign(deltaY))) == 1)
    Position(sign(deltaX), 0)
    # uniformChoice(rng, [Position(sign(deltaX), 0), Position(0, sign(deltaY))])
  else
    Position(sign(deltaX), sign(deltaY))  
  end
end

function unitVector(object1::NamedTuple, object2::NamedTuple, @nospecialize(state::NamedTuple))::Position
  position1 = object1.origin
  position2 = object2.origin
  unitVector(position1, position2, state)
end

function unitVector(@nospecialize(object::NamedTuple), position::Position, @nospecialize(state::NamedTuple))::Position
  unitVector(object.origin, position, state)
end

function unitVector(position::Position, @nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::Position
  unitVector(position, object.origin, state)
end

function unitVector(position::Position, @nospecialize(state::NamedTuple))::Position
  unitVector(Position(0,0), position, state)
end 

function displacement(position1::Position, position2::Position, @nospecialize(state=nothing))::Position
  Position(floor(Int, position2.x - position1.x), floor(Int, position2.y - position1.y))
end

function displacement(cell1::Cell, cell2::Cell, @nospecialize(state=nothing))::Position
  displacement(cell1.position, cell2.position)
end

function adjacent(position1::Position, position2::Position, @nospecialize(state=nothing))::Bool
  displacement(position1, position2) in [Position(0,1), Position(1, 0), Position(0, -1), Position(-1, 0)]
end

function adjacent(cell1::Cell, cell2::Cell, @nospecialize(state=nothing))::Bool
  adjacent(cell1.position, cell2.position)
end

function adjacent(cell::Cell, cells::AbstractArray, @nospecialize(state=nothing))
  length(filter(x -> adjacent(cell, x), cells)) != 0
end

function adjacentObjs(@nospecialize(obj::NamedTuple), @nospecialize(state::NamedTuple))
  filter(o -> adjacent(o.origin, obj.origin) && (obj.id != o.id), state.scene.objects)
end

function adjacentObjsDiag(@nospecialize(obj::NamedTuple), @nospecialize(state::NamedTuple))
  filter(o -> adjacentDiag(o.origin, obj.origin) && (obj.id != o.id), state.scene.objects)
end

function adjacentDiag(position1::Position, position2::Position, @nospecialize(state=nothing))
  displacement(position1, position2) in [Position(0,1), Position(1, 0), Position(0, -1), Position(-1, 0),
                                         Position(1,1), Position(1, -1), Position(-1, 1), Position(-1, -1)]
end

function rotate(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :render, map(x -> Cell(rotate(x.position), x.color), new_object.render))
  new_object
end

function rotate(position::Position, @nospecialize(state=nothing))::Position
  Position(-position.y, position.x)
 end

function rotateNoCollision(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(rotate(object), state) && isFree(rotate(object), object, state)) ? rotate(object) : object
end

function move(position1::Position, position2::Position, @nospecialize(state=nothing))
  Position(position1.x + position2.x, position1.y + position2.y)
end

function move(position::Position, cell::Cell, @nospecialize(state=nothing))
  Position(position.x + cell.position.x, position.y + cell.position.y)
end

function move(cell::Cell, position::Position, @nospecialize(state=nothing))
  Position(position.x + cell.position.x, position.y + cell.position.y)
end

function move(@nospecialize(object::NamedTuple), position::Position, @nospecialize(state=nothing))
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, move(object.origin, position))
  new_object
end

function move(@nospecialize(object::NamedTuple), x::Int, y::Int, @nospecialize(state=nothing))::NamedTuple
  move(object, Position(x, y))                          
end

# ----- begin left/right move ----- #

function moveLeft(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  move(object, Position(-1, 0))                          
end

function moveRight(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  move(object, Position(1, 0))                          
end

function moveUp(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  move(object, Position(0, -1))                          
end

function moveDown(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  move(object, Position(0, 1))                          
end

# ----- end left/right move ----- #

function moveNoCollision(@nospecialize(object::NamedTuple), position::Position, @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(move(object, position), state) && isFree(move(object, position.x, position.y), object, state)) ? move(object, position.x, position.y) : object 
end

function moveNoCollision(@nospecialize(object::NamedTuple), x::Int, y::Int, @nospecialize(state::NamedTuple))
  (isWithinBounds(move(object, x, y), state) && isFree(move(object, x, y), object, state)) ? move(object, x, y) : object 
end

# ----- begin left/right moveNoCollision ----- #

function moveLeftNoCollision(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(move(object, -1, 0), state) && isFree(move(object, -1, 0), object, state)) ? move(object, -1, 0) : object 
end

function moveRightNoCollision(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(move(object, 1, 0), state) && isFree(move(object, 1, 0), object, state)) ? move(object, 1, 0) : object 
end

function moveUpNoCollision(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(move(object, 0, -1), state) && isFree(move(object, 0, -1), object, state)) ? move(object, 0, -1) : object 
end

function moveDownNoCollision(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple
  (isWithinBounds(move(object, 0, 1), state) && isFree(move(object, 0, 1), object, state)) ? move(object, 0, 1) : object 
end

# ----- end left/right moveNoCollision ----- #

function moveWrap(@nospecialize(object::NamedTuple), position::Position, @nospecialize(state::NamedTuple))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, position.x, position.y, state))
  new_object
end

function moveWrap(cell::Cell, position::Position, @nospecialize(state::NamedTuple))
  moveWrap(cell.position, position.x, position.y, state)
end

function moveWrap(position::Position, cell::Cell, @nospecialize(state::NamedTuple))
  moveWrap(cell.position, position, state)
end

function moveWrap(@nospecialize(object::NamedTuple), x::Int, y::Int, @nospecialize(state::NamedTuple))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, x, y, state))
  new_object
end

function moveWrap(position1::Position, position2::Position, @nospecialize(state::NamedTuple))::Position
  moveWrap(position1, position2.x, position2.y, state)
end

function moveWrap(position::Position, x::Int, y::Int, @nospecialize(state::NamedTuple))::Position
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
    # # println("hello")
    # # println(Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE))
    Position((position.x + x + GRID_SIZE_X) % GRID_SIZE_X, (position.y + y + GRID_SIZE_Y) % GRID_SIZE_Y)
  else
    Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE)
  end

end

# ----- begin left/right moveWrap ----- #

function moveLeftWrap(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, -1, 0, state))
  new_object
end
  
function moveRightWrap(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 1, 0, state))
  new_object
end

function moveUpWrap(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 0, -1, state))
  new_object
end

function moveDownWrap(@nospecialize(object::NamedTuple), @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 0, 1, state))
  new_object
end

# ----- end left/right moveWrap ----- #

function randomPositions(GRID_SIZE, n::Int, @nospecialize(state=nothing))::Array{Position}
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
    nums = uniformChoice([0:(GRID_SIZE_X * GRID_SIZE_Y - 1);], n, state)
    map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)    
  else
    nums = uniformChoice([0:(GRID_SIZE * GRID_SIZE - 1);], n, state)
    map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
  end
end

function distance(position1::Position, position2::Position, @nospecialize(state=nothing))::Int
  abs(position1.x - position2.x) + abs(position1.y - position2.y)
end

function distance(object1::NamedTuple, object2::NamedTuple, @nospecialize(state=nothing))::Int
  position1 = object1.origin
  position2 = object2.origin
  distance(position1, position2)
end

function distance(@nospecialize(object::NamedTuple), position::Position, @nospecialize(state=nothing))::Int
  distance(object.origin, position)
end

function distance(position::Position, @nospecialize(object::NamedTuple), @nospecialize(state=nothing))::Int
  distance(object.origin, position)
end

function closest(@nospecialize(object::NamedTuple), type::Symbol, @nospecialize(state::NamedTuple))::Position
  objects_of_type = filter(obj -> (obj.type == type) && (obj.alive), state.scene.objects)
  if length(objects_of_type) == 0
    object.origin
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    filter(obj -> distance(object, obj) == min_distance, objects_of_type)[1].origin
  end
end

function mapPositions(constructor, GRID_SIZE, filterFunction, args, @nospecialize(state=nothing))::Union{NamedTuple, Array{<:NamedTuple}}
  map(pos -> constructor(args..., pos), filter(filterFunction, allPositions(GRID_SIZE)))
end

function allPositions(GRID_SIZE, @nospecialize(state=nothing))
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
    nums = [0:(GRID_SIZE_X * GRID_SIZE_Y - 1);]
    map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)
  else
    nums = [0:(GRID_SIZE * GRID_SIZE - 1);]
    map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
  end
end

function updateOrigin(@nospecialize(object::NamedTuple), new_origin::Position, @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, new_origin)
  new_object
end

function updateAlive(@nospecialize(object::NamedTuple), new_alive::Bool, @nospecialize(state=nothing))::NamedTuple
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :alive, new_alive)
  new_object
end

function nextLiquid(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple 
  # # println("nextLiquid")
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  new_object = deepcopy(object)
  if object.origin.y != GRID_SIZE_Y - 1 && isFree(move(object.origin, Position(0, 1)), state)
    new_object = update_nt(new_object, :origin, move(object.origin, Position(0, 1)))
  else
    leftHoles = filter(pos -> (pos.y == object.origin.y + 1)
                               && (pos.x < object.origin.x)
                               && isFree(pos, state), allPositions(state))
    rightHoles = filter(pos -> (pos.y == object.origin.y + 1)
                               && (pos.x > object.origin.x)
                               && isFree(pos, state), allPositions(state))
    if (length(leftHoles) != 0) || (length(rightHoles) != 0)
      if (length(leftHoles) == 0)
        closestHole = closest(object, rightHoles)
        if isFree(move(closestHole, Position(0, -1)), move(object.origin, Position(1, 0)), state)
          new_object = update_nt(new_object, :origin, move(object.origin, unitVector(object, move(closestHole, Position(0, -1)), state), state))
        end
      elseif (length(rightHoles) == 0)
        closestHole = closest(object, leftHoles)
        if isFree(move(closestHole, Position(0, -1)), move(object.origin, Position(-1, 0)), state)
          new_object = update_nt(new_object, :origin, move(object.origin, unitVector(object, move(closestHole, Position(0, -1)), state)))                      
        end
      else
        closestLeftHole = closest(object, leftHoles)
        closestRightHole = closest(object, rightHoles)
        if distance(object.origin, closestLeftHole) > distance(object.origin, closestRightHole)
          if isFree(move(object.origin, Position(1, 0)), move(closestRightHole, Position(0, -1)), state)
            new_object = update_nt(new_object, :origin, move(object.origin, unitVector(new_object, move(closestRightHole, Position(0, -1)), state)))
          elseif isFree(move(closestLeftHole, Position(0, -1)), move(object.origin, Position(-1, 0)), state)
            new_object = update_nt(new_object, :origin, move(object.origin, unitVector(new_object, move(closestLeftHole, Position(0, -1)), state)))
          end
        else
          if isFree(move(closestLeftHole, Position(0, -1)), move(object.origin, Position(-1, 0)), state)
            new_object = update_nt(new_object, :origin, move(object.origin, unitVector(new_object, move(closestLeftHole, Position(0, -1)), state)))
          elseif isFree(move(object.origin, Position(1, 0)), move(closestRightHole, Position(0, -1)), state)
            new_object = update_nt(new_object, :origin, move(object.origin, unitVector(new_object, move(closestRightHole, Position(0, -1)), state)))
          end
        end
      end
    end
  end
  new_object
end

function nextSolid(@nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::NamedTuple 
  # # println("nextSolid")
  new_object = deepcopy(object)
  if (isWithinBounds(move(object, Position(0, 1)), state) && reduce(&, map(x -> isFree(x, object, state), map(cell -> move(cell.position, Position(0, 1)), render(object)))))
    new_object = update_nt(new_object, :origin, move(object.origin, Position(0, 1)))
  end
  new_object
end

function closest(@nospecialize(object::NamedTuple), positions::Array{Position}, @nospecialize(state=nothing))::Position
  closestDistance = sort(map(pos -> distance(pos, object.origin), positions))[1]
  closest = filter(pos -> distance(pos, object.origin) == closestDistance, positions)[1]
  closest
end

function isFree(start::Position, stop::Position, @nospecialize(state::NamedTuple))::Bool 
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  translated_start = GRID_SIZE_X * start.y + start.x 
  translated_stop = GRID_SIZE_X * stop.y + stop.x
  if translated_start < translated_stop
    ordered_start = translated_start
    ordered_end = translated_stop
  else
    ordered_start = translated_stop
    ordered_end = translated_start
  end
  nums = [ordered_start:ordered_end;]
  reduce(&, map(num -> isFree(Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), state), nums))
end

function isFree(start::Position, stop::Position, @nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))::Bool 
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  translated_start = GRID_SIZE_X * start.y + start.x 
  translated_stop = GRID_SIZE_X * stop.y + stop.x
  if translated_start < translated_stop
    ordered_start = translated_start
    ordered_end = translated_stop
  else
    ordered_start = translated_stop
    ordered_end = translated_start
  end
  nums = [ordered_start:ordered_end;]
  reduce(&, map(num -> isFree(Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), object, state), nums))
end

function isFree(position::Position, @nospecialize(object::NamedTuple), @nospecialize(state::NamedTuple))
  length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, 
  render((objects=filter(obj -> obj.id != object.id , state.scene.objects), background=state.scene.background)))) == 0
end

function isFree(@nospecialize(object::NamedTuple), @nospecialize(orig_object::NamedTuple), @nospecialize(state::NamedTuple))::Bool
  reduce(&, map(x -> isFree(x, orig_object, state), map(cell -> cell.position, render(object))))
end

function allPositions(@nospecialize(state::NamedTuple))
  GRID_SIZE = state.GRID_SIZEHistory[0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  nums = [1:GRID_SIZE_X*GRID_SIZE_Y - 1;]
  map(num -> Position(num % GRID_SIZE_X, floor(Int, num / GRID_SIZE_X)), nums)
end

function unfold(A, @nospecialize(state=nothing))
  V = []
  for x in A
      for elt in x
        push!(V, elt)
      end
  end
  V
end

# interpretutils.jl 

function sub(aex::AExpr, (x, v))
  # print("SUb")
  # # # # # # @showaex
  # # # # # # @showx
  # # # # # # @showv
  arr = [aex.head, aex.args...]
  # next(x) = interpret(x, Γ)
  if (x isa AExpr) && ([x.head, x.args...] == arr)  
    v 
  else
    MLStyle.@match arr begin
      [:fn, args, body]                                       => AExpr(:fn, args, sub(body, x => v))
      [:if, c, t, e]                                          => AExpr(:if, sub(c, x => v), sub(t, x => v), sub(e, x => v))
      [:assign, a1, a2]                                       => AExpr(:assign, a1, sub(a2, x => v))
      [:list, args...]                                        => AExpr(:list, map(arg -> sub(arg, x => v), args)...)
      [:typedecl, args...]                                    => AExpr(:typedecl, args...)
      [:let, args...]                                         => AExpr(:let, map(arg -> sub(arg, x => v), args)...)      
      # [:case, args...] => compilecase(expr, data)            
      # [:typealias, args...] => compiletypealias(expr, data)      
      [:lambda, args, body]                                   => AExpr(:fn, args, sub(body, x => v))
      [:call, f, args...]                                     => AExpr(:call, f, map(arg -> sub(arg, x => v) , args)...)      
      [:field, o, fieldname]                                  => AExpr(:field, sub(o, x => v), fieldname)
      [:object, args...]                                      => AExpr(:object, args...)
      [:on, event, update]                                    => AExpr(:on, sub(event, x => v), sub(update, x => v))
      [args...]                                               => throw(AutumnError(string("Invalid AExpr Head: ", expr.head)))
      _                                                       => error("Could not sub $arr")
    end
  end
end

sub(aex::Symbol, (x, v)) = aex == x ? v : aex
sub(aex, (x, v)) = aex # aex is a value

const Environment = NamedTuple
empty_env() = NamedTuple()
std_env() = empty_env()

"Produces new environment Γ' s.t. `Γ(x) = v` (and everything else unchanged)"
# update(@nospecialize(Γ::NamedTuple), x::Symbol, v) = merge(Γ, NamedTuple{(x,)}((v,)))

function update(@nospecialize(Γ::NamedTuple), x::Symbol, @nospecialize(v)) 
  merge(Γ, NamedTuple{(x,)}((v,)))
end

# primitive function handling 
prim_to_func = Dict(:+ => +,
                    :- => -,
                    :* => *,
                    :/ => /,
                    :& => &,
                    :! => !,
                    :| => |,
                    :> => >,
                    :>= => >=,
                    :< => <,
                    :<= => <=,
                    :(==) => ==,
                    :% => %,
                    :!= => !=)

isprim(f) = f in keys(prim_to_func)
# primapl(f, x...) = (prim_to_func[f](x[1:end-1]...), x[end])

function primapl(f, x, @nospecialize(Γ::NamedTuple)) 
  prim_to_func[f](x), Γ
end

function primapl(f, x1, x2, @nospecialize(Γ::NamedTuple))
  prim_to_func[f](x1, x2), Γ
end

lib_to_func = Dict(:Position => Position,
                   :Cell => Cell,
                   :Click => Click,
                   :render => render, 
                   :occurred => occurred,
                   :uniformChoice => uniformChoice, 
                   :min => min,
                   :isWithinBounds => isWithinBounds, 
                   :clicked => clicked, 
                   :objClicked => objClicked, 
                   :intersects => intersects, 
                   :addObj => addObj, 
                   :removeObj => removeObj, 
                   :updateObj => updateObj,
                   :filter_fallback => filter_fallback,
                   :adjPositions => adjPositions,
                   :isWithinBounds => isWithinBounds,
                   :isFree => isFree, 
                   :rect => rect, 
                   :unitVector => unitVector, 
                   :displacement => displacement, 
                   :adjacent => adjacent, 
                   :adjacentObjs => adjacentObjs, 
                   :adjacentObjsDiag => adjacentObjsDiag,
                   :rotate => rotate, 
                   :rotateNoCollision => rotateNoCollision, 
                   :move => move, 
                   :moveLeft => moveLeft, 
                   :moveRight => moveRight, 
                   :moveUp => moveUp, 
                   :moveDown => moveDown, 
                   :moveNoCollision => moveNoCollision, 
                   :moveLeftNoCollision => moveLeftNoCollision, 
                   :moveRightNoCollision => moveRightNoCollision, 
                   :moveDownNoCollision => moveDownNoCollision, 
                   :moveUpNoCollision => moveUpNoCollision, 
                   :moveWrap => moveWrap, 
                   :moveLeftWrap => moveLeftWrap,
                   :moveRightWrap => moveRightWrap, 
                   :moveUpWrap => moveUpWrap, 
                   :moveDownWrap => moveDownWrap, 
                   :randomPositions => randomPositions, 
                   :distance => distance,
                   :closest => closest, 
                   :mapPositions => mapPositions, 
                   :allPositions => allPositions, 
                   :updateOrigin => updateOrigin, 
                   :updateAlive => updateAlive, 
                   :nextLiquid => nextLiquid, 
                   :nextSolid => nextSolid,
                   :unfold => unfold,
                   :prev => prev
                )
islib(f) = f in keys(lib_to_func)

# library function handling 
function libapl(f, args, @nospecialize(Γ::NamedTuple))
  # # # println("libapl")
  # # # @showf 
  # # # @showargs 

  if f == :clicked && (length(args) == 0)
    interpret(f, Γ)
  elseif f == :clicked
    lib_to_func[f](interpret(:click, Γ)[1], args..., Γ.state), Γ
  else
    has_function_arg = false
    for arg in args 
      if (arg isa AbstractArray) && (length(arg) == 2) && (arg[1] isa AExpr || arg[1] isa Symbol) && (arg[2] isa AExpr || arg[2] isa Symbol)
        has_function_arg = true
      end
    end
  
    if !has_function_arg && (f != :updateObj)
      # # # # println("CHECK HERE")
      # # # # @showf
      # # # # @showargs
      # # # # @showkeys(Γ.state)
      # # # @showargs 
      lib_to_func[f](map(x -> interpret(x, Γ)[1], args)..., Γ.state), Γ    
    else
      if f == :updateObj 
        interpret_updateObj(args, Γ)
      else # f == :removeObj 
        interpret_removeObj(args, Γ)
      end
    end
  end
end

julia_lib_to_func = Dict(:get => get, 
                         :map => map,
                         :filter => filter,
                         :first => first,
                         :last => last,
                         :in => in, 
                         :intersect => intersect,
                         :length => length,
                         :sign => sign,
                         :vcat => vcat, 
                         :count => count,)
isjulialib(f) = f in keys(julia_lib_to_func)

function julialibapl(f, args, @nospecialize(Γ::NamedTuple))
  # # # println("julialibapl")
  # # # @showf 
  # # # @showargs 
  if !(f in [:map, :filter])
    julia_lib_to_func[f](args...), Γ
  elseif f == :map 
    interpret_julia_map(args, Γ)
  elseif f == :filter 
    interpret_julia_filter(args, Γ)
  end
end

function interpret(aex::AExpr, @nospecialize(Γ::NamedTuple))
  arr = [aex.head, aex.args...]
  # # # # # # println()
  # # # # # # println("Env:")
  # display(Γ)
  # # # # # # @showarr 
  # next(x) = interpret(x, Γ)
  isaexpr(x) = x isa AExpr
  t = MLStyle.@match arr begin
    [:if, c, t, e]                                              => let (v, Γ2) = interpret(c, Γ) 
                                                                       if v == true
                                                                         interpret(t, Γ2)
                                                                       else
                                                                         interpret(e, Γ2)
                                                                       end
                                                                   end
    [:assign, x, v::AExpr] && if v.head == :initnext end        => interpret_init_next(x, v, Γ)
    [:assign, x, v::Union{AExpr, Symbol}]                       => let (v2, Γ_) = interpret(v, Γ)
                                                                    # # # @showv 
                                                                    # # # @showx
                                                                     interpret(AExpr(:assign, x, v2), Γ_)
                                                                   end
    [:assign, x, v]                                             => let
                                                                     # # @showx 
                                                                     # # @showv 
                                                                     if x in fieldnames(typeof(Γ))
                                                                      # # println("here") 
                                                                      # # @showΓ[x]
                                                                     end
                                                                     # # @showupdate(Γ, x, v)[x]
                                                                     # # println("returning")
                                                                     (aex, update(Γ, x, v)) 
                                                                   end
    [:list, args...]                                            => interpret_list(args, Γ)
    [:typedecl, args...]                                        => (aex, Γ)
    [:let, args...]                                             => interpret_let(args, Γ) 
    [:lambda, args...]                                          => (args, Γ)
    [:fn, args...]                                              => (args, Γ)
    [:call, f, arg1] && if isprim(f) end                        => let (new_arg, Γ2) = interpret(arg1, Γ)
                                                                     primapl(f, new_arg, Γ2)
                                                                   end
                                                                    
    [:call, f, arg1, arg2] && if isprim(f) end                  => let (new_arg1, Γ2) = interpret(arg1, Γ)
                                                                       (new_arg2, Γ2) = interpret(arg2, Γ2)
                                                                       primapl(f, new_arg1, new_arg2, Γ2)
                                                                   end
    [:call, f, args...] && if f == :prev && args != [:obj] end  => interpret(AExpr(:call, Symbol(string(f, uppercasefirst(string(args[1])))), :state), Γ)
    [:call, f, args...] && if islib(f) end                      => interpret_lib(f, args, Γ)
    [:call, f, args...] && if isjulialib(f) end                 => interpret_julia_lib(f, args, Γ)
    [:call, f, args...] && if f in keys(Γ[:object_types]) end   => interpret_object_call(f, args, Γ)
    [:call, f, args...]                                         => interpret_call(f, args, Γ)
     
    [:field, x, fieldname]                                      => interpret_field(x, fieldname, Γ)
    [:object, args...]                                          => interpret_object(args, Γ)
    [:on, args...]                                              => interpret_on(args, Γ)
    [args...]                                                   => error(string("Invalid AExpr Head: ", aex.head))
    _                                                           => error("Could not interpret $arr")
  end
  # # # # # # println("FINSIH", arr)
  # # # # # @show(t)
  # # println("T[2]")
  # # @showt[2]
  t
end

function interpret(x::Symbol, @nospecialize(Γ::NamedTuple))
  # # # @showkeys(Γ)
  if x == Symbol("false")
    false, Γ
  elseif x == Symbol("true")
    true, Γ
  elseif x == :clicked 
    interpret(AExpr(:call, :occurred, :click), Γ)
  elseif x in keys(Γ[:object_types])
    x, Γ
  elseif x in keys(Γ)
    # # # @showeval(:($(Γ).$(x)))
    eval(:($(Γ).$(x))), Γ
  else
    error("Could not interpret $x")
  end
end

# if x is not an AExpr or a Symbol, it is just a value (return it)
function interpret(x, @nospecialize(Γ::NamedTuple))
  if x isa BigInt 
    (convert(Int, x), Γ)
  else
    (x, Γ)
  end
end 

function interpret_list(args, @nospecialize(Γ::NamedTuple))
  new_list = []
  for arg in args
    new_arg, Γ = interpret(arg, Γ)
    push!(new_list, new_arg)
  end
  new_list, Γ
end

function interpret_lib(f, args, @nospecialize(Γ::NamedTuple))
  # # # println("INTERPRET_LIB")
  # # # @showf 
  # # # @showargs 
  new_args = []
  for arg in args 
    new_arg, Γ = interpret(arg, Γ)
    push!(new_args, new_arg)
  end
  # # # @shownew_args
  libapl(f, new_args, Γ)
end

function interpret_julia_lib(f, args, @nospecialize(Γ::NamedTuple))
  # println("INTERPRET_JULIA_LIB")
  # @show f 
  # @show args
  new_args = []
  for arg in args 
    # # # @showarg 
    new_arg, Γ = interpret(arg, Γ)
    # # # @shownew_arg 
    # # # @showΓ
    push!(new_args, new_arg)
  end
  # @show new_args 
  julialibapl(f, new_args, Γ)
end

function interpret_field(x, f, @nospecialize(Γ::NamedTuple))
  # # # # println("INTERPRET_FIELD")
  # # # # @showkeys(Γ)
  # # # # @showx 
  # # # # @showf 
  val, Γ2 = interpret(x, Γ)
  if val isa NamedTuple 
    (val[f], Γ2)
  else
    (eval(:($(val).$(f))), Γ2)
  end
end

function interpret_let(args::AbstractArray, @nospecialize(Γ::NamedTuple))
  Γ2 = Γ
  if length(args) > 0
    for arg in args[1:end-1] # all lines in let except last
      v2, Γ2 = interpret(arg, Γ2)
    end
  
    if args[end] isa AExpr
      if args[end].head == :assign # all assignments are global; no return value 
        v2, Γ2 = interpret(args[end], Γ2)
        (AExpr(:let, args...), Γ2)
      else # return value is AExpr   
        v2, Γ2 = interpret(args[end], Γ2)
        (v2, Γ)
      end
    else # return value is not AExpr
      (interpret(args[end], Γ2)[1], Γ)
    end
  else
    AExpr(:let, args...), Γ2
  end
end

function interpret_call(f, params, @nospecialize(Γ::NamedTuple))
  func, Γ = interpret(f, Γ)
  func_args = func[1]
  func_body = func[2]

  # construct environment 
  Γ2 = Γ
  if func_args isa AExpr 
    for i in 1:length(func_args.args)
      param_name = func_args.args[i]
      param_val, Γ2 = interpret(params[i], Γ2)
      Γ2 = update(Γ2, param_name, param_val)
    end
  elseif func_args isa Symbol
    param_name = func_args
    param_val, Γ2 = interpret(params[1], Γ2)
    Γ2 = update(Γ2, param_name, param_val)
  else
    error("Could not interpret $(func_args)")
  end
  # # # # # @showtypeof(Γ2)
  # evaluate func_body in environment 
  v, Γ2 = interpret(func_body, Γ2)
  
  # return value and original environment, except with state updated 
  Γ = update(Γ, :state, update(Γ.state, :objectsCreated, Γ2.state.objectsCreated))
  # # # # # println("DONE")
  (v, Γ)
end

function interpret_object_call(f, args, @nospecialize(Γ::NamedTuple))
  # # # # # println("BEFORE")
  # # # # # @showΓ.state.objectsCreated 
  new_state = update(Γ.state, :objectsCreated, Γ.state.objectsCreated + 1)
  Γ = update(Γ, :state, new_state)

  origin, Γ = interpret(args[end], Γ)
  object_repr = (origin=origin, type=f, alive=true, changed=false, id=Γ.state.objectsCreated)

  Γ2 = Γ
  fields = Γ2.object_types[f][:fields]
  for i in 1:length(fields)
    field_name = fields[i].args[1]
    field_value, Γ2 = interpret(args[i], Γ2)
    object_repr = update(object_repr, field_name, field_value)
    Γ2 = update(Γ2, field_name, field_value)
  end

  render, Γ2 = interpret(Γ.object_types[f][:render], Γ2)
  render = render isa AbstractArray ? render : [render]
  object_repr = update(object_repr, :render, render)
  # # # # # println("AFTER")
  # # # # # @showΓ.state.objectsCreated 
  (object_repr, Γ)  
end

function interpret_init_next(var_name, var_val, @nospecialize(Γ::NamedTuple))
  # # # println("INTERPRET INIT NEXT")
  init_func = var_val.args[1]
  next_func = var_val.args[2]

  Γ2 = Γ
  if !(var_name in keys(Γ2)) # variable not initialized; use init clause
    # # # println("HELLO")
    # initialize var_name
    var_val, Γ2 = interpret(init_func, Γ2)
    Γ2 = update(Γ2, var_name, var_val)

    # construct history variable in state 
    new_state = update(Γ2.state, Symbol(string(var_name, "History")), Dict())
    Γ2 = update(Γ2, :state, new_state)

    # construct prev function 
    _, Γ2 = interpret(AExpr(:assign, Symbol(string(:prev, uppercasefirst(string(var_name)))), parseautumn("""(fn (state) (get (.. state $(string(var_name, "History"))) (- (.. state time) 1) -1))""")), Γ2) 

  elseif Γ.state.time > 0 # variable initialized; use next clause if simulation time > 0  
    # update var_val 
    var_val = Γ[var_name]
    if var_val isa Array 
      changed_val = filter(x -> x.changed, var_val) # values changed by on-clauses
      unchanged_val = filter(x -> !x.changed, var_val) # values unchanged by on-clauses; apply default behavior
      # # # # @showvar_val 
      # # # # @showchanged_val 
      # # # # @showunchanged_val
      # replace (prev var_name) or var_name with unchanged_val 
      modified_next_func = sub(next_func, AExpr(:call, :prev, var_name) => unchanged_val)
      modified_next_func = sub(modified_next_func, var_name => unchanged_val)
      # # # # println("HERE I AM ONCE AGAIN")
      # # # # @showΓ.state.objectsCreated
      default_val, Γ = interpret(modified_next_func, Γ)
      # # # # @showdefault_val 
      # # # # println("HERE I AM ONCE AGAIN 2")
      # # # # @showΓ.state.objectsCreated
      final_val = map(o -> update(o, :changed, false), filter(obj -> obj.alive, vcat(changed_val, default_val)))
    else # variable is not an array
      if var_name in keys(Γ[:on_clauses])
        events = Γ[:on_clauses][var_name]
      else
        events = []
      end
      changed = false 
      for e in events 
        v, Γ = interpret(e, Γ)
        if v == true 
          changed = true
        end
      end
      if !changed 
        final_val, Γ = interpret(next_func, Γ)
      else
        final_val = var_val
      end
    end
    Γ2 = update(Γ, var_name, final_val)
  end
  (AExpr(:assign, var_name, var_val), Γ2)
end

function interpret_object(args, @nospecialize(Γ::NamedTuple))
  object_name = args[1]
  object_fields = args[2:end-1]
  object_render = args[end]

  # construct object creation function 
  object_tuple = (render=object_render, fields=object_fields)
  Γ = update(Γ, :object_types, update(Γ[:object_types], object_name, object_tuple))
  (AExpr(:object, args...), Γ)
end

function interpret_on(args, @nospecialize(Γ::NamedTuple))
  # # # println("INTERPRET ON")
  event = args[1]
  update_ = args[2]
  # # # @showevent 
  # # # @showupdate_
  Γ2 = Γ
  if Γ2.state.time == 0 
    if update_.head == :assign
      var_name = update_.args[1]
      if !(var_name in keys(Γ2[:on_clauses]))
        Γ2 = update(Γ2, :on_clauses, update(Γ2[:on_clauses], var_name, [event]))
      else
        Γ2 = update(Γ2, :on_clauses, update(Γ2[:on_clauses], var_name, vcat(event, Γ2[:on_clauses][var_name])))
      end
    elseif update_.head == :let 
      assignments = update_.args
      if length(assignments) > 0 
        if (assignments[end] isa AExpr) && (assignments[end].head == :assign)
          for a in assignments 
            var_name = a.args[1]
            if !(var_name in keys(Γ2[:on_clauses]))
              Γ2 = update(Γ2, :on_clauses, update(Γ2[:on_clauses], var_name, [event]))
            else
              Γ2 = update(Γ2, :on_clauses, update(Γ2[:on_clauses], var_name, vcat(event, Γ2[:on_clauses][var_name])))
            end
          end
        end
      end
    else
      error("Could not interpret $(update_)")
    end
  else
    # println("ON CLAUSE")
    # # @showevent 
    # # # @showupdate_  
    # @show repr(event)
    e, Γ2 = interpret(event, Γ2) 
    # # @showe 
    # @show update_
    if e == true
      # println("EVENT IS TRUE!") 
      t = interpret(update_, Γ2)
      # println("WHAT ABOUT HERE")
      # @show t[2]
      Γ3 = t[2]
      # # println("hi")
      Γ2 = Γ3
    end
  end
  (AExpr(:on, args...), Γ2)
end

# evaluate updateObj on arguments that include functions 
function interpret_updateObj(args, @nospecialize(Γ::NamedTuple))
  # println("MADE IT!")
  Γ2 = Γ
  numFunctionArgs = count(x -> x == true, map(arg -> (arg isa AbstractArray) && (length(arg) == 2) && (arg[1] isa AExpr || arg[1] isa Symbol) && (arg[2] isa AExpr || arg[2] isa Symbol), args))
  if numFunctionArgs == 1
    list, Γ2 = interpret(args[1], Γ2)
    map_func = args[2]

    # # # # @showlist 
    # # # # @showmap_func

    new_list = []
    for item in list 
      # # # # # println("PRE=PLS WORK")
      # # # # # @showΓ2.state.objectsCreated      
      new_item, Γ2 = interpret(AExpr(:call, map_func, item), Γ2)
      # # # # # println("PLS WORK")
      # # # # # @showΓ2.state.objectsCreated
      push!(new_list, new_item)
    end
    new_list, Γ2
  elseif numFunctionArgs == 2
    list, Γ2 = interpret(args[1], Γ2)
    map_func = args[2]
    filter_func = args[3]

    # @show list 
    # @show map_func


    new_list = []
    for item in list 
      pred, Γ2 = interpret(AExpr(:call, filter_func, item), Γ2)
      if pred == true 
        # println("PRED TRUE!")
        # @show item 
        new_item, Γ2 = interpret(AExpr(:call, map_func, item), Γ2)
        push!(new_list, new_item)
      else
        # println("PRED FALSE!")
        # @show item 
        push!(new_list, item)
      end
    end
    # @show new_list 
    new_list, Γ2
  elseif numFunctionArgs == 0
    obj = args[1]
    field_string = args[2]
    new_value = args[3]
    new_obj = update(obj, Symbol(field_string), new_value)
    new_obj = update(new_obj, :changed, true)

    # update render 
    object_type = Γ[:object_types][obj.type]
    
    Γ3 = Γ2
    fields = object_type[:fields]
    for i in 1:length(fields)
      field_name = fields[i].args[1]
      field_value = new_obj[field_name]
      Γ3 = update(Γ3, field_name, field_value)
    end
  
    render, Γ3 = interpret(Γ.object_types[obj.type][:render], Γ3)
    render = render isa AbstractArray ? render : [render]
    new_obj = update(new_obj, :render, render)

    new_obj, Γ2
    # Γ2 = update(Γ2, :state, update(Γ2.state, :objectsCreated, Γ2.state.objectsCreated + 1))
  else
    error("Could not interpret updateObj")
  end
end

function interpret_removeObj(args, @nospecialize(Γ::NamedTuple))
  # # # @showargs
  list, Γ = interpret(args[1], Γ)
  func = args[2]
  new_list = []
  for item in list
    pred, Γ = interpret(AExpr(:call, func, item), Γ) 
    if pred == false 
      push!(new_list, item)
    else
      new_item = update(update(item, :alive, false), :changed, true)
      push!(new_list, new_item)
    end
  end
  new_list, Γ
end

function interpret_julia_map(args, @nospecialize(Γ::NamedTuple))
  new_list = []
  map_func = args[1]
  list, Γ = interpret(args[2], Γ)
  for arg in list  
    new_arg, Γ = interpret(AExpr(:call, map_func, arg), Γ)
    push!(new_list, new_arg)
  end
  new_list, Γ
end

function interpret_julia_filter(args, @nospecialize(Γ::NamedTuple))
  new_list = []
  filter_func = args[1]
  list, Γ = interpret(args[2], Γ)
  for arg in list
    v, Γ = interpret(AExpr(:call, filter_func, arg), Γ)
    if v == true 
      push!(new_list, interpret(arg, Γ)[1])
    end
  end
  new_list, Γ
end

# interpret.jl

function interpret_program(aex, @nospecialize(Γ::NamedTuple))
  aex.head == :program || error("Must be a program aex")
  for line in aex.args
    v, Γ = interpret(line, Γ)
  end
  return aex, Γ
end

function start(aex::AExpr, rng=Random.GLOBAL_RNG)
  aex.head == :program || error("Must be a program aex")
  env = (object_types=empty_env(), 
         on_clauses=empty_env(),
         left=false, 
         right=false,
         up=false,
         down=false,
         click=nothing, 
         state=(time=0, objectsCreated=0, rng=rng, scene=empty_env()))
  lines = aex.args 

  # reorder program lines
  grid_params_and_object_type_lines = filter(l -> !(l.head in [:assign, :on]), lines) # || (l.head == :assign && l.args[1] in [:GRID_SIZE, :background])
  initnext_lines = filter(l -> l.head == :assign && (l.args[2] isa AExpr && l.args[2].head == :initnext), lines)
  lifted_lines = filter(l -> l.head == :assign && (!(l.args[2] isa AExpr) || l.args[2].head != :initnext), lines) # GRID_SIZE and background here
  on_clause_lines = filter(l -> l.head == :on, lines)

  reordered_lines_temp = vcat(grid_params_and_object_type_lines, 
                              initnext_lines, 
                              on_clause_lines, 
                              lifted_lines)

  reordered_lines = vcat(grid_params_and_object_type_lines, 
                         on_clause_lines, 
                         initnext_lines, 
                         lifted_lines)

  # add prev functions and variable history to state for lifted variables 
  for line in lifted_lines
    var_name = line.args[1] 
    # construct history variable in state
    new_state = update(env.state, Symbol(string(var_name, "History")), Dict())
    env = update(env, :state, new_state)

    # construct prev function 
    _, env = interpret(AExpr(:assign, Symbol(string(:prev, uppercasefirst(string(var_name)))), parseautumn("""(fn () (get (.. state $(string(var_name, "History"))) (- (.. state time) 1) $(var_name)))""")), env) 
  end

  # add background to scene 
  background_assignments = filter(l -> l.args[1] == :background, lifted_lines)
  background = background_assignments != [] ? background_assignments[end].args[2] : "#ffffff00"
  env = update(env, :state, update(env.state, :scene, update(env.state.scene, :background, background)))


  # initialize scene.objects 
  env = update(env, :state, update(env.state, :scene, update(env.state.scene, :objects, [])))

  # initialize lifted variables
  env = update(env, :lifted, empty_env()) 
  for line in lifted_lines
    var_name = line.args[1]
    env = update(env, :lifted, update(env.lifted, var_name, line.args[2])) 
    if var_name in [:GRID_SIZE, :background]
      env = update(env, var_name, interpret(line.args[2], env)[1])
    end
  end 

  new_aex = AExpr(:program, reordered_lines_temp...) # try interpreting the init_next's before on for the first time step (init)
  # # @show new_aex
  aex_, env_ = interpret_program(new_aex, env)

  # update state (time, histories, scene)
  env_ = update_state(env_)

  AExpr(:program, reordered_lines...), env_
end

function step(aex::AExpr, @nospecialize(env::NamedTuple), user_events=(click=nothing, left=false, right=false, down=false, up=false))::NamedTuple
  # update env with user event 
  for user_event in keys(user_events)
    if !isnothing(user_events[user_event])
      env = update(env, user_event, user_events[user_event])
    end
  end

  aex_, env_ = interpret_program(aex, env)

  # update state (time, histories, scene) + reset user_event
  env_ = update_state(env_)
  
  env_
end

"""Update the history variables, scene, and time fields of env_.state"""
function update_state(@nospecialize(env_::NamedTuple))
  # reset user events 
  for user_event in [:left, :right, :up, :down]
    env_ = update(env_, user_event, false)
  end
  env_ = update(env_, :click, nothing)

  # add updated variable values to history
  for key in filter(sym -> occursin("History", string(sym)), keys(env_.state))
    var_name = Symbol(replace(string(key), "History" => ""))
    env_.state[key][env_.state.time] = env_[var_name]
  end

  # update lifted variables 
  for var_name in keys(env_.lifted)
    env_ = update(env_, var_name, interpret(env_.lifted[var_name], env_)[1])
  end

  # update scene.objects 
  new_scene_objects = []
  for key in keys(env_)
    if !(key in [:state, :object_types, :on_clauses, :lifted]) && ((env_[key] isa NamedTuple && (:id in keys(env_[key]))) || (env_[key] isa AbstractArray && (length(env_[key]) > 0) && (env_[key][1] isa NamedTuple)))
      object_val = env_[key]
      if object_val isa AbstractArray 
        push!(new_scene_objects, object_val...)
      else
        push!(new_scene_objects, object_val)
      end
    end
  end
  env_ = update(env_, :state, update(env_.state, :scene, update(env_.state.scene, :objects, new_scene_objects)))

  # update time 
  new_state = update(env_.state, :time, env_.state.time + 1)
  env_ = update(env_, :state, new_state)
end

function interpret_over_time(aex::AExpr, iters, user_events=[])::NamedTuple
  new_aex, env_ = start(aex)
  if user_events == []
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_)
    end
  else
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_, user_events[i])
    end
  end
  env_
end

function interpret_over_time_observations(aex::AExpr, iters, user_events=[])
  scenes = []
  new_aex, env_ = start(aex)
  push!(scenes, env_.state.scene)
  if user_events == []
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_)
      push!(scenes, env_.state.scene)
    end
  else
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_, user_events[i])
      push!(scenes, env_.state.scene)
    end
  end
  map(s -> render_scene(s), scenes)
end

end
