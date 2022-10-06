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
using Statistics
using SExpressions
using Distributions: Categorical
using Setfield
using JLD 
using Dates
using Colors
using Images
import Base.min

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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  @show env_.current_var_values
  grid_size = env_.current_var_values[:GRID_SIZE] isa AbstractArray ? env_.current_var_values[:GRID_SIZE] : [[env_.current_var_values[:GRID_SIZE], env_.current_var_values[:GRID_SIZE]]]
  background = env_.state.scene.background
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], interpret(AExpr(:call, :renderScene, env_.state.scene), env_)[1])
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

const log_dir = "saved_test_traces_user_study/" # "/Users/riadas/Documents/urop/today_temp/CausalDiscovery.jl/saved_test_traces/"

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

# abstract type Object end
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
    push!(choices, ("adjPositions", [:(genPosition($(environment)))]))
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

# scene.jl 

"""
Example Use:
> rng = MersenneTwister(0)
> image = render(generatescene_objects(rng))
> save("scene.png", colorview(RGBA, image))
> println(parsescene_image(image))

Example Use for Inference
> rng = MersenneTwister(0)
> generate_and_save_random_scene_inf(rng) # saves image as scene2.png
"""

"""Generate and save random scene as scene.png"""
function generate_and_save_random_scene(rng)
  image = render(generatescene_objects(rng))
  save("scene.png", colorview(RGBA, image))
end

"""Generate and save random scene as scene_inf.png"""
function generate_and_save_random_scene_inf(rng)
  image = render_inf(generatescene_objects_inf(rng))
  save("scene_inf.png", colorview(RGBA, image))
end

# ----- NEW: render functions for inference ----- # 

"""Input: list of objects, where each object is a tuple (shape, color, position)"""
function render_inf(objects; gridsize=16, transparent=false)
  background = "white"
  image = [RGBA(1.0, 0.0, 0.0, 1.0) for x in 1:gridsize, y in 1:gridsize]
  for object in objects
    center_x = object[3][1]
    center_y = object[3][2]
    shape = object[1]
    color = rgb(object[2])
    for shape_position in shape
      shape_x, shape_y = shape_position
      x = center_x + shape_x
      y = center_y + shape_y
      if (x > 0) && (x <= gridsize) && (y > 0) && (y <= gridsize) # only render in-bound pixel positions
        if transparent 
          if image[y, x] == RGBA(1.0, 0.0, 0.0, 1.0)
            image[y, x] = RGBA(color.r, color.g, color.b, 0.6)
          else
            new_alpha = image[y,x].alpha + 0.6 - image[y,x].alpha * 0.6
            image[y, x] = RGBA((image[y,x].alpha * image[y,x].r + 0.6*(1 - image[y,x].alpha)*color.r)/new_alpha,
                               (image[y,x].alpha * image[y,x].g + 0.6*(1 - image[y,x].alpha)*color.g)/new_alpha,
                               (image[y,x].alpha * image[y,x].b + 0.6*(1 - image[y,x].alpha)*color.b)/new_alpha,
                              new_alpha)
          end  
        else
          image[y, x] = RGBA(color.r, color.g, color.b, 0.6)
        end
      end
    end
  end
  for x in 1:gridsize
    for y in 1:gridsize
      if image[x, y] == RGBA(1.0, 0.0, 0.0, 1.0)
        image[x, y] = rgb(background)
      end
    end
  end
  image
end

function generatescene_objects_inf(rng=Random.GLOBAL_RNG; gridsize::Int=16)
  types, objects, background, gridsize = generatescene_objects(rng, gridsize=gridsize)
  formatted_objects = []
  for object in sort(objects, by=x->x.type.id)
    push!(formatted_objects, (object.type.shape, object.type.color, object.position))
  end
  formatted_objects
end

# ----- define colors and color-related functions ----- # 

colors = ["red", "yellow", "green", "blue"]
backgroundcolors = ["white", "black"]

"""Euclidean distance between two RGB/RGBA colors"""
function colordist(color1, color2)
  (color1.r - color2.r)^2 + (color1.g - color2.g)^2 + (color1.b - color2.b)^2 
end

"""CSS string color name from RGB color"""
function colorname(r::RGB)
  rgbs = vcat(keys(rgb_to_colorname)...)
  colordists = map(x -> colordist(r, x), rgbs)
  minidx = findall(x -> x == minimum(colordists), colordists)[1]
  rgb_key = rgbs[minidx]
  rgb_to_colorname[rgb_key]
end

"""CSS string color name from RGBA color"""
function colorname(rgba::RGBA)
  rgb = RGB(rgba.r, rgba.g, rgba.b)
  colorname(rgb)
end

"""RGB value from CSS string color name"""
function rgb(colorname)
  colorname_to_rgb[colorname]
end

rgb_to_colorname = Dict([
  (colorant"red", "red"),
  (colorant"yellow", "yellow"),
  (colorant"green", "green"),
  (colorant"blue", "blue"),
  (colorant"white", "white"),
  (colorant"black", "black")
]);

colorname_to_rgb = Dict([
  ("red", colorant"red"),
  ("yellow", colorant"yellow"),
  ("green", colorant"green"),
  ("blue", colorant"blue"),
  ("white", colorant"white"),
  ("black", colorant"black")
])

# ----- end define colors and color-related functions ----- # 

# ----- define general utils ----- #

"""Compute neighbor positions of given shape"""
function neighbors(shape::AbstractArray)
  neighborPositions = vcat(map(pos -> neighbors(pos), shape)...)
  unique(filter(pos -> !(pos in shape), neighborPositions))
end

"""Compute neighbor positions of given position"""
function neighbors(position)
  x = position[1]
  y = position[2]
  [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
end

# ----- end define general utils ----- #

# ----- define functions related to generative model over scenes ----- #
mutable struct ObjType
  shape::AbstractArray
  color::String
  custom_fields::AbstractArray
  id::Int
end

mutable struct Obj
  type::ObjType
  position::Tuple{Int, Int}
  custom_field_values::AbstractArray
  id::Int
end

function render_from_cells(cells, gridsize) 
  background = "white"
  image = [RGBA(1.0, 0.0, 0.0, 1.0) for x in 1:gridsize, y in 1:gridsize]
  for cell in cells 
    x = cell.position.x 
    y = cell.position.y 
    if (x > 0) && (x <= gridsize) && (y > 0) && (y <= gridsize)
      if image[y, x] == RGBA(1.0, 0.0, 0.0, 1.0)
        image[y, x] = RGBA(color.r, color.g, color.b, 0.6)
      else
        new_alpha = image[y,x].alpha + 0.6 - image[y,x].alpha * 0.6
        image[y, x] = RGBA((image[y,x].alpha * image[y,x].r + 0.6*(1 - image[y,x].alpha)*color.r)/new_alpha,
                           (image[y,x].alpha * image[y,x].g + 0.6*(1 - image[y,x].alpha)*color.g)/new_alpha,
                           (image[y,x].alpha * image[y,x].b + 0.6*(1 - image[y,x].alpha)*color.b)/new_alpha,
                          new_alpha)
      end  
    end
  end 

  for x in 1:gridsize
    for y in 1:gridsize
      if image[x, y] == RGBA(1.0, 0.0, 0.0, 1.0)
        image[x, y] = rgb(background)
      end
    end
  end

  image
end

"""Produce image from types_and_objects scene representation"""
function render(types_and_objects)
  types, objects, background, gridsize = types_and_objects
  image = [RGBA(1.0, 0.0, 0.0, 1.0) for x in 1:gridsize, y in 1:gridsize]
  println("""
  # (program
  #   (= GRID_SIZE $(gridsize))
  #   (= background "$(background)")
  #   $(join(map(t -> "(object ObjType$(t.id) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) "$(t.color)")""", t.shape), " "))))", types), "\n  "))

  #   $((join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))...)

  #   $((join(map(obj -> """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) (Position $(obj.position[1] - 1) $(obj.position[2] - 1))) (prev obj$(obj.id))))""", objects), "\n  ")))
  # )
  # """)
  for object in objects
    center_x = object.position[1]
    center_y = object.position[2]
    type = object.type
    color = rgb(object.type.color)
    for shape_position in type.shape
      shape_x, shape_y = shape_position
      x = center_x + shape_x
      y = center_y + shape_y
      if (x > 0) && (x <= gridsize) && (y > 0) && (y <= gridsize) # only render in-bound pixel positions
        if image[y, x] == RGBA(1.0, 0.0, 0.0, 1.0)
          image[y, x] = RGBA(color.r, color.g, color.b, 0.6)
        else
          new_alpha = image[y,x].alpha + 0.6 - image[y,x].alpha * 0.6
          image[y, x] = RGBA((image[y,x].alpha * image[y,x].r + 0.6*(1 - image[y,x].alpha)*color.r)/new_alpha,
                             (image[y,x].alpha * image[y,x].g + 0.6*(1 - image[y,x].alpha)*color.g)/new_alpha,
                             (image[y,x].alpha * image[y,x].b + 0.6*(1 - image[y,x].alpha)*color.b)/new_alpha,
                            new_alpha)
        end  
      end
    end
  end
  for x in 1:gridsize
    for y in 1:gridsize
      if image[x, y] == RGBA(1.0, 0.0, 0.0, 1.0)
        image[x, y] = rgb(background)
      end
    end
  end
  image
end

function program_string(types_and_objects)
  types, objects, background, gridsize = types_and_objects 
  """ 
  (program
    (= GRID_SIZE $(gridsize))
    (= background "$(background)")
    $(join(map(t -> "(object ObjType$(t.id) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) "$(t.color)")""", t.shape), " "))))", types), "\n  "))

    $((join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))...)

    $((join(map(obj -> """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) (Position $(obj.position[1] - 1) $(obj.position[2] - 1))) (prev obj$(obj.id))))""", objects), "\n  ")))
  )
  """
end

function program_string_synth(types_and_objects)
  types, objects, background, gridsize = types_and_objects 
  """ 
  (program
    (= GRID_SIZE $(gridsize))
    (= background "$(background)")
    $(join(map(t -> "(object ObjType$(t.id) $(join(map(tuple -> "(: $(tuple[1]) $(tuple[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) $(t.custom_fields == [] ? """ "$(t.color)" """ : "color"))""", t.shape), " "))))", types), "\n  "))

    $((join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))...)

    $((join(map(obj -> """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) $(join(map(v -> """ "$(v)" """, obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2]))) (prev obj$(obj.id))))""", objects), "\n  ")))
  )
  """
end

function program_string_synth_update_rule(types_and_objects)
  types, objects, background, gridsize = types_and_objects 
  """ 
  (program
    (= GRID_SIZE $(gridsize isa Int ? gridsize : "(list $(gridsize[1]) $(gridsize[2]))"))
    (= background "$(background)")
    $(join(map(t -> "(object ObjType$(t.id) $(join(map(tuple -> "(: $(tuple[1]) $(tuple[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) $(t.custom_fields == [] ? """ "$(t.color)" """ : "color"))""", t.shape), " "))))", types), "\n  "))

    $((join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))...)

    $((join(map(obj -> obj.position != (-1, -1) ? """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) $(join(map(v -> """ "$(v)" """, obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2]))) (prev obj$(obj.id))))"""
                                                : """(= obj$(obj.id) (initnext (removeObj (ObjType$(obj.type.id) $(join(map(v -> """ "$(v)" """, obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2])))) (prev obj$(obj.id))))""", objects), "\n  ")))
  )
  """
end

function program_string_synth_standard_groups(object_decomposition)
  object_types, object_mapping, background, gridsize = object_decomposition

  start_objects = sort(filter(obj -> obj != nothing, [object_mapping[i][1] for i in 1:length(collect(keys(object_mapping)))]), by=(x -> x.id))

  start_type_mapping = Dict()
  num_objects_with_type = Dict()

  for type in object_types
    start_type_mapping[type.id] = sort(filter(obj -> obj.type.id == type.id, start_objects), by=(x -> x.id))
    num_objects_with_type[type.id] = count(obj_id -> filter(x -> !isnothing(x), object_mapping[obj_id])[1].type.id == type.id, collect(keys(object_mapping)))
  end
  

  start_constants_and_lists = vcat(filter(l -> length(l) == 1 && (num_objects_with_type[l[1].type.id] == 1), map(k -> start_type_mapping[k], collect(keys(start_type_mapping))))..., 
                                   filter(l -> length(l) > 1 || ((length(l) == 1) && (num_objects_with_type[l[1].type.id] > 1)), map(k -> start_type_mapping[k], collect(keys(start_type_mapping)))))
  start_constants_and_lists = sort(start_constants_and_lists, by=x -> x isa Array ? x[1].id : x.id)

  other_list_types = filter(k -> (length(start_type_mapping[k]) == 0) || (length(start_type_mapping[k]) == 1 && num_objects_with_type[k] == 1), sort(collect(keys(start_type_mapping))))

  """ 
  (program
    (= GRID_SIZE $(gridsize isa Int ? gridsize : "(list $(gridsize[1]) $(gridsize[2]))"))
    (= background "$(background)")
    $(join(map(t -> "(object ObjType$(t.id) $(join(map(tuple -> "(: $(tuple[1]) $(tuple[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) $(filter(tuple -> tuple[1] == "color", t.custom_fields) == [] ? """ "$(t.color)" """ : "color"))""", t.shape), " "))))", object_types), "\n  "))

    $((join(map(obj -> obj isa Array ? """(: addedObjType$(obj[1].type.id)List (List ObjType$(obj[1].type.id)))""" 
                                     : """(: obj$(obj.id) ObjType$(obj.type.id))""", start_constants_and_lists), "\n  "))...)

    $((join(map(k -> """(: addedObjType$(k)List (List ObjType$(k)))""", other_list_types), "\n  "))...)

    $((join(map(obj -> obj isa Array ? """(= addedObjType$(obj[1].type.id)List (initnext (list $(join(map(x -> "(ObjType$(x.type.id) $(join(map(v -> v isa String ? """ "$(v)" """ : "$(v)", x.custom_field_values), " ")) (Position $(x.position[1]) $(x.position[2])))", obj), " "))) (prev addedObjType$(obj[1].type.id)List)))"""
                                     : """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) $(join(map(v -> v isa String ? """ "$(v)" """ : "$(v)", obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2]))) (prev obj$(obj.id))))""", start_constants_and_lists), "\n  ")))
    
    $(join(map(k -> """(= addedObjType$(k)List (initnext (list) (prev addedObjType$(k)List)))""", other_list_types), "\n  ")...)
  )
  """
end

function program_string_synth_standard_groups_multi_trace_reset(object_decomposition, stop_time)
  object_types, object_mapping, background, gridsize = object_decomposition

  start_objects = sort(filter(obj -> obj != nothing, [object_mapping[i][1] for i in 1:length(collect(keys(object_mapping)))]), by=(x -> x.id))

  start_type_mapping = Dict()
  num_objects_with_type = Dict()

  for type in object_types
    start_type_mapping[type.id] = sort(filter(obj -> obj.type.id == type.id, start_objects), by=(x -> x.id))
    num_objects_with_type[type.id] = count(obj_id -> filter(x -> !isnothing(x), object_mapping[obj_id])[1].type.id == type.id, collect(keys(object_mapping)))
  end
  

  start_constants_and_lists = vcat(filter(l -> length(l) == 1 && (num_objects_with_type[l[1].type.id] == 1), map(k -> start_type_mapping[k], collect(keys(start_type_mapping))))..., 
                                   filter(l -> length(l) > 1 || ((length(l) == 1) && (num_objects_with_type[l[1].type.id] > 1)), map(k -> start_type_mapping[k], collect(keys(start_type_mapping)))))


  removed_object_ids = sort(filter(id -> !isnothing(object_mapping[id][1]) && isnothing(object_mapping[id][stop_time]), collect(keys(object_mapping))))
  removed_objects = map(id -> object_mapping[id][1], removed_object_ids)

  start_constants_and_lists = filter(x -> x != [] && !(x in removed_objects), map(obj -> obj isa AbstractArray ? filter(o -> !(o in removed_objects), obj) : obj, start_constants_and_lists))
  start_constants_and_lists = sort(start_constants_and_lists, by=x -> x isa Array ? x[1].id : x.id)

  added_back_object_ids = sort(filter(id -> !isnothing(object_mapping[id][stop_time + 1]) && isnothing(object_mapping[id][stop_time]), collect(keys(object_mapping))))
  added_back_objects = map(id -> object_mapping[id][stop_time + 1], added_back_object_ids)

  other_list_types = filter(k -> (length(start_type_mapping[k]) == 0) || (length(start_type_mapping[k]) == 1 && num_objects_with_type[k] == 1), sort(collect(keys(start_type_mapping))))
  
  start_constants_and_lists_strings = []
  for obj in start_constants_and_lists 
    if !(obj isa Array) 
      pos = obj.position
      obj_str = """(updateObj (prev obj$(obj.id)) "origin" (Position $(pos[1]) $(pos[2])))"""
      for i in 1:length(obj.type.custom_fields)
        field = obj.type.custom_fields[i]
        val = obj.custom_field_values[i]
        obj_str = """(updateObj $(obj_str) "$(field[1])" $(val isa String ? """ "$(val)" """ : val))"""
      end
      obj_str = """(= obj$(obj.id) $(obj_str))"""
      push!(start_constants_and_lists_strings, obj_str)
    else
      individual_object_updates = []
      for o in obj 
        pos = o.position
        obj_str = """(updateObj (prev obj) "origin" (Position $(pos[1]) $(pos[2])))"""
        for i in 1:length(o.type.custom_fields)
          field = o.type.custom_fields[i]
          val = o.custom_field_values[i]
          obj_str = """(updateObj $(obj_str) "$(field[1])" $(val isa String ? """ "$(val)" """ : val))"""
        end
        obj_str = """(= addedObjType$(o.type.id)List (updateObj addedObjType$(o.type.id)List (--> obj $(obj_str)) (--> obj (== (.. obj id) $(o.id)))))"""
        push!(individual_object_updates, obj_str)
      end
      o_ids = map(o -> o.id, obj)
      removal_str = """(= addedObjType$(obj[1].type.id)List (removeObj addedObjType$(obj[1].type.id)List (--> obj (! (in (.. obj id) (list $(join(o_ids, " "))))))))"""
      push!(individual_object_updates, removal_str)
      push!(start_constants_and_lists_strings, join(individual_object_updates, "\n"))
    end
  end

  # add removed objects 
  removed_object_strings = []
  for obj in added_back_objects
    if obj.type.id in other_list_types # singleton (constant) object
      obj_str = """(updateObj (prev obj$(obj.id)) "alive" true)"""
      pos = obj.position 
      obj_str = """(updateObj $(obj_str) "origin" (Position $(pos[1]) $(pos[2])))"""
      for i in 1:length(obj.type.custom_fields)
        field = obj.type.custom_fields[i]
        val = obj.custom_field_values[i]
        obj_str = """(updateObj $(obj_str) "$(field[1])" $(val isa String ? """ "$(val)" """ : val))"""
      end

      push!(removed_object_strings, """(= obj$(obj.id) $(obj_str))""")
    else # object is in a list
      pos = obj.position 
      obj_str = """(ObjType$(obj.type.id) $(join(map(v -> v isa String ? """ "$(v)" """ : v, obj.custom_field_values), " ")) (Position $(pos[1]) $(pos[2])))"""
      push!(removed_object_strings, """(= addedObjType$(obj.type.id)List (addObj addedObjType$(obj.type.id)List $(obj_str)))""")
    end
  end

  """
  $(join(start_constants_and_lists_strings, "\n\n"))

    $(join(map(k -> """(= addedObjType$(k)List (removeObj addedObjType$(k)List (--> obj true)))""", other_list_types), "\n  ")...)

    $(join(removed_object_strings, "\n\n"))
  """
end

function program_string_synth_custom_groups(object_decomposition, group_structure)
  object_types, object_mapping, background, gridsize = object_decomposition

  objects = group_structure["objects"] # list of single-variable objects (ordered by id)
  group_dict = group_structure["groups"] # dictionary from group ids to list of ordered object ids 
  group_ids = sort(collect(keys(group_dict)))

  """ 
  (= GRID_SIZE $(gridsize))
  (= background "$(background)")
  $(join(map(t -> "(object ObjType$(t.id) $(join(map(tuple -> "(: $(tuple[1]) $(tuple[2]))", t.custom_fields), " ")) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) $(t.custom_fields == [] ? """ "$(t.color)" """ : "color"))""", t.shape), " "))))", object_types), "\n  "))

  $(join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))
  $(join(map(group_id -> """(: group$(group_id) (List ObjType$(filter(x -> !isnothing(x), object_mapping[group_dict[group_id][1]])[1].type.id)))""", group_ids), "\n  "))

  $(join(map(obj -> """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) $(join(map(v -> """ "$(v)" """, obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2]))) (prev obj$(obj.id))))""", objects), "\n  "))
  $(join(map(group_id -> """(= group$(group_id) (initnext (list $(join(map(obj -> """(ObjType$(obj.type.id) $(join(map(v -> """ "$(v)" """, obj.custom_field_values), " ")) (Position $(obj.position[1]) $(obj.position[2])))""" , filter(x -> !isnothing(x), map(obj_id -> object_mapping[obj_id][1], group_dict[group_id]))), " "))) (prev group$(group_id))))""", group_ids), "\n  "))
  """
end

function generatescene_program(rng=Random.GLOBAL_RNG; gridsize::Int=16)
  types_and_objects = generatescene_objects(rng, gridsize=gridsize)
  program_string(types_and_objects)
end

function generatescene_objects(rng=Random.GLOBAL_RNG; gridsize::Int=16)
  background = backgroundcolors[rand(1:length(backgroundcolors))]
  numObjects = rand(rng, 1:20)
  numTypes = rand(rng, 1:min(numObjects, 5))
  types = [] # each type has form (list of position tuples, index in types list)::Tuple{Array{Tuple{Int, Int}}, Int}
  
  objectPositions = [(rand(rng, 1:gridsize), rand(rng, 1:gridsize)) for x in 1:numObjects]
  objects = [] # each object has form (type, position tuple, color, index in objects list)

  for type in 1:numTypes
    renderSize = rand(rng, 1:5)
    shape = [(0,0)]
    while length(shape) != renderSize
      boundaryPositions = neighbors(shape)
      push!(shape, boundaryPositions[rand(rng, 1:length(boundaryPositions))])
    end
    color = colors[rand(rng, 1:length(colors))]
    
    # generate custom fields
    custom_fields = []
    num_fields = rand(0:2)
    for i in 1:num_fields
      push!(custom_fields, ("field$(i)", rand(["Int", "Bool"])))
    end

    push!(types, ObjType(shape, color, custom_fields, length(types) + 1))
  end

  for i in 1:numObjects
    objPosition = objectPositions[i]
    objType = types[rand(rng, 1:length(types))]

    # generate custom field values
    custom_fields = objType.custom_fields
    custom_field_values = map(field -> field[2] == "Int" ? rand(1:3) : rand(["true", "false"]), custom_fields)

    push!(objects, Obj(objType, objPosition, custom_field_values, length(objects) + 1))    
  end
  (types, objects, background, gridsize)
end

# ----- end functions related to generative model over scenes ----- # 

# ----- define functions related to scene parsing -----

function parsescene_image_singlecell(image)
  dimImage = size(image)[1]
  background = count(x -> x == "white", map(colorname, image)) > count(x -> x == "black", map(colorname, image)) ? "white" : "black"
  colors = []
  objects = []
  for y in 1:dimImage
    for x in 1:dimImage
      color = colorname(image[y, x])
      if color != "white"
        if !(color in colors)
          push!(colors, color)
        end

        push!(objects, (x - 1, y - 1, (color, findall(x -> x == color, colors)[1]), length(objects) + 1))
      end
    end
  end

  """
  (program
    (= GRID_SIZE $(dimImage))
    (= background "$(background)")

    $(join(map(color -> """(object ObjType$(findall(x -> x == color, colors)[1]) (Cell 0 0 "$(color)"))""", colors), "\n  "))

    $(join(map(obj -> """(: obj$(obj[4]) ObjType$(obj[3][2]))""", objects), "\n  "))

    $(join(map(obj -> """(= obj$(obj[4]) (initnext (ObjType$(obj[3][2]) (Position $(obj[1]) $(obj[2]))) (prev obj$(obj[4]))))""", objects), "\n  "))
  )
  """
end

function color_contiguity(image, pos1, pos2)
  image[pos1[1], pos1[2]] == image[pos2[1], pos2[2]]
end

function parsescene_image(image; color=true)
  dimImage = size(image)[1]
  background = count(x -> x == "white", map(colorname, image)) > count(x -> x == "black", map(colorname, image)) ? "white" : "black"
  objectshapes = []
  colored_positions = map(ci -> (ci.I[2], ci.I[1]), findall(color -> color != "white", map(colorname, image)))
  visited = []
  for position in colored_positions
    if !(position in visited)
      objectshape = []
      q = Queue{Any}()
      enqueue!(q, position)
      while !isempty(q)
        pos = dequeue!(q)
        push!(objectshape, pos)
        push!(visited, pos)
        pos_neighbors = neighbors(pos)
        for n in pos_neighbors
          if (n in colored_positions) && !(n in visited) && (color ? color_contiguity(image, n, pos) : true) 
            enqueue!(q, n)
          end
        end
      end
      push!(objectshapes, objectshape)
    end
  end

  types = []
  objects = []
  for objectshape in objectshapes
    objectcolors = map(pos -> colorname(image[pos[2], pos[1]]), objectshape)
    
    translated = map(pos -> dimImage * (pos[2] - 1)+ (pos[1] - 1), objectshape)
    translated = length(translated) % 2 == 0 ? translated[1:end-1] : translated
    centerPos = objectshape[findall(x -> x == median(translated), translated)[1]]
    translatedShape = map(pos -> (pos[1] - centerPos[1], pos[2] - centerPos[2]), objectshape)
    translatedShapeWithColors = [(translatedShape[i], objectcolors[i]) for i in 1:length(translatedShape)]

    push!(types, (translatedShapeWithColors, length(types) + 1))
    push!(objects, (centerPos, length(types), length(objects) + 1))
  end

  """
  (program
    (= GRID_SIZE $(dimImage))
    (= background "$(background)")

    $(join(map(t -> """(object ObjType$(t[2]) (list $(join(map(cell -> """(Cell $(cell[1][1]) $(cell[1][2]) "$(cell[2])")""", t[1]), " ")))""", types), "\n  "))

    $(join(map(obj -> """(: obj$(obj[3]) ObjType$(obj[2]))""", objects), "\n  "))
 
    $(join(map(obj -> """(= obj$(obj[3]) (initnext (ObjType$(obj[2]) (Position $(obj[1][1] - 1) $(obj[1][2] - 1))) (prev obj$(obj[3]))))""", objects), "\n  "))
  )
  """
end

function color_contiguity_autumn(position_to_color, pos1, pos2)
  length(intersect(position_to_color[pos1], position_to_color[pos2])) > 0
end

function parsescene_autumn(render_output::AbstractArray, dim::Int=16, background::String="white"; color=true)
  colors = unique(map(cell -> cell.color, render_output))

  objectshapes = []
  for c in colors   
    colored_positions = sort(map(cell -> (cell.position.x, cell.position.y), filter(cell -> cell.color == c, render_output)))
    if length(unique(colored_positions)) == length(colored_positions) # no overlapping objects of the same color 
      visited = []
      for position in colored_positions
        if !(position in visited)
          objectshape = []
          q = Queue{Any}()
          enqueue!(q, position)
          while !isempty(q)
            pos = dequeue!(q)
            push!(objectshape, (pos, c))
            push!(visited, pos)
            pos_neighbors = neighbors(pos)
            for n in pos_neighbors
              if (n in colored_positions) && !(n in visited) 
                enqueue!(q, n)
              end
            end
          end
          push!(objectshapes, objectshape)
        end
      end
    else # overlapping objects of the same color; parse each cell as a different object
      for position in colored_positions 
        push!(objectshapes, [(position, c)])
      end
    end
  end  
  # # @show objectshapes

  types = []
  objects = []
  # # # @show length(objectshapes)
  for objectshape_with_color in objectshapes
    objectcolor = objectshape_with_color[1][2]
    objectshape = map(o -> o[1], objectshape_with_color)
    # # # @show objectcolor 
    # # # @show objectshape
    translated = map(pos -> dim * pos[2] + pos[1], objectshape)
    translated = length(translated) % 2 == 0 ? translated[1:end-1] : translated # to produce a single median
    centerPos = objectshape[findall(x -> x == median(translated), translated)[1]]
    translatedShape = unique(map(pos -> (pos[1] - centerPos[1], pos[2] - centerPos[2]), sort(objectshape)))

    if !((translatedShape, objectcolor) in map(type -> (type.shape, type.color) , types))
      push!(types, ObjType(translatedShape, objectcolor, [], length(types) + 1))
      push!(objects, Obj(types[length(types)], centerPos, [], length(objects) + 1))
    else
      type_id = findall(type -> (type.shape, type.color) == (translatedShape, objectcolor), types)[1]
      push!(objects, Obj(types[type_id], centerPos, [], length(objects) + 1))
    end
  end

  # combine types with the same shape but different colors
  # println("INSIDE REGULAR PARSER")
  # # @show types 
  # # @show objects
  (types, objects) = combine_types_with_same_shape(types, objects)

  (types, objects, background, dim)
end

function parsescene_autumn_given_types(render_output::AbstractArray, override_types::AbstractArray, dim::Int=16, background::String="white"; color=true)
  (standard_types, objects, _, _) = parsescene_autumn(render_output, dim, background, color=color)
  # println("OBJECTS")
  # println(objects)
  # println("OBJECT_TYPES")
  # println(standard_types)
  # extract multi-cellular types that do not appear in override_types
  old_override_types = deepcopy(override_types)
  # # @show override_types 
  # # @show standard_types
  types_to_ungroup = filter(s_type -> (length(s_type.shape) > 1), standard_types)

  # extract single-cell types 
  grouped_type_colors = map(grouped_type -> length(grouped_type.custom_fields) == 0 ? [grouped_type.color] : grouped_type.custom_fields[1][3], types_to_ungroup)
  composition_types = map(colors -> (filter(type -> (length(type.shape) == 1) && (( (length(type.custom_fields) == 0) && (length(intersect(colors, [type.color])) > 0)) 
                                                                                 || (length(type.custom_fields) > 0) && (length(intersect(colors, type.custom_fields[1][3])) > 0)), standard_types), 
                                     filter(type -> (length(type.shape) == 1) && (( (length(type.custom_fields) == 0) && (length(intersect(colors, [type.color])) > 0)) 
                                                                                 || (length(type.custom_fields) > 0) && (length(intersect(colors, type.custom_fields[1][3])) > 0)), override_types)), 
                          grouped_type_colors)

  # only consider types to ungroup that have single-celled types of the same color
  # # @show types_to_ungroup 
  # # @show composition_types
  # # @show length(types_to_ungroup)
  # # @show length(composition_types)
  remove_types_to_ungroup = []
  for i in 1:length(composition_types)
    if composition_types[i] == ([], [])
      # println("WHAT")
      # println(map(type -> type.id, types_to_ungroup))
      # println(types_to_ungroup[i].id)
      push!(remove_types_to_ungroup, types_to_ungroup[i])
      # println(length(types_to_ungroup))
    end
  end
  types_to_ungroup = filter(t -> !(t.id in map(type -> type.id, remove_types_to_ungroup)), types_to_ungroup)
  composition_types = filter(types -> types != ([], []), composition_types)

  # println("READY")
  # # @show types_to_ungroup 
  # # @show composition_types

  if (length(types_to_ungroup) == 0) # no types to try ungrouping
    new_objects = objects 
    new_types = standard_types
  else # exist types to ungroup
    # # @show types_to_ungroup
    new_objects = filter(obj -> !(obj.type.id in map(type -> type.id, types_to_ungroup)), objects)
    new_types = standard_types
    # println("HELLO 1")
    # # # @show new_types
    for grouped_type_id in 1:length(types_to_ungroup)
      grouped_type = types_to_ungroup[grouped_type_id]
      composition_types_group = composition_types[grouped_type_id]

      if length(grouped_type.custom_fields) == 0
        filter!(type -> type.id != grouped_type.id, new_types) # remove grouped type from new_types
        # println("-------------------> LOOK AT ME")
        # # # @show new_types
        # determine composition type
        # # @show grouped_type_id
        # # @show composition_types_group
        if length(composition_types_group[1]) > 0 # composition type present in standard types
          composition_type = composition_types_group[1][1]
        else # composition type present in override types only 
          composition_type = composition_types_group[2][1]
          composition_type.id = grouped_type.id # switch the composition type's id to the grouped type's id, since we're eliminating the grouped type
          push!(new_types, composition_type)
        end
        
        objects_to_ungroup = filter(obj -> obj.type.id == grouped_type.id, objects)
        # # # @show objects_to_ungroup
        for object in objects_to_ungroup
          for pos in object.type.shape
            if composition_type.custom_fields == []
              push!(new_objects, Obj(composition_type, (pos[1] + object.position[1], pos[2] + object.position[2]), [], length(new_objects) + length(objects)))
            else
              push!(new_objects, Obj(composition_type, (pos[1] + object.position[1], pos[2] + object.position[2]), [object.type.color], length(new_objects) + length(objects)))
            end
          end
        end
      else # grouped object supports multiple colors
        # println("HERE I AM 2")
        colors = deepcopy(grouped_type.custom_fields[1][3])
        # println(colors)
        for color in colors 
          # println("HERE I AM 3")
          # # @show composition_types_group
          # println(vcat(vcat(map(types_list -> map(type -> vcat(type.color, (length(type.custom_fields) == 0 ? [] : type.custom_fields[1][3])...), types_list), composition_types_group)...)...))
          if color in vcat(vcat(map(types_list -> map(type -> vcat(type.color, (length(type.custom_fields) == 0 ? [] : type.custom_fields[1][3])...), types_list), composition_types_group)...)...)
            # println("HERE I AM")
            # # @show grouped_type
            # # @show color
            if length(composition_types_group[1]) > 0 # composition type present in standard types
              in_standard_bool = true
              type = composition_types_group[1][1]
              if color in vcat(type.color, (type.custom_fields == [] ? [] : type.custom_fields[1][3])...)
                composition_type = type
              end
            else # composition type present in override types only
              in_standard_bool = false
              type = composition_types_group[2][1]
              if color in vcat(type.color, (type.custom_fields == [] ? [] : type.custom_fields[1][3])...)
                composition_type = type
              end
            end
            
            # remove color from grouped type OR remove object if all colors have been removed
            if length(grouped_type.custom_fields[1][3]) == 2
              filter!(c -> c != color, grouped_type.custom_fields[1][3])
              type.color = color
              # println("WHY 1")
              # println(grouped_type)
            elseif length(grouped_type.custom_fields[1][3]) == 1
              filter!(type -> type.id != grouped_type.id, new_types) # remove object if all colors have been eliminated
            else
              filter!(c -> c != color, grouped_type.custom_fields[1][3])
            end
            # println("-----> HERE 2")
            # println(new_types)

            if !(in_standard_bool) 
              composition_type.id = length(new_types) + length(override_types) + 1
              push!(new_types, composition_type)
              # println("----> HERE")
              # println(composition_type)
            end
            
            objects_to_ungroup = filter(obj -> (obj.type.id == grouped_type.id) && 
                                               (((obj.custom_field_values == []) && obj.color == color) || (obj.custom_field_values == [color])), objects)
            # println("OBJECTS TO UNGROUP")
            # println(grouped_type.id)
            # println(objects_to_ungroup)
            for object in objects_to_ungroup
              for pos in object.type.shape
                if composition_type.custom_fields == []
                  push!(new_objects, Obj(composition_type, (pos[1] + object.position[1], pos[2] + object.position[2]), [], length(new_objects) + length(objects)))
                else
                  push!(new_objects, Obj(composition_type, (pos[1] + object.position[1], pos[2] + object.position[2]), object.custom_field_values != [] ? object.custom_field_values : [object.type.color], length(new_objects) + length(objects)))
                end
              end
            end
          
          else # color not in custom_fields
            # println("ADD BACK?")
            # println(objects)
            # println(grouped_type.id)
            # println(filter(obj -> (obj.type.id == grouped_type.id) && (obj.custom_field_values == [color]), objects))
            # add previously removed objects back
            push!(new_objects, filter(obj -> (obj.type.id == grouped_type.id) && (obj.custom_field_values == [color]), objects)...)
          end
        end
      end
    end
    # println("HELLO 3")
    # # # @show new_types
    # # # @show new_objects 
    # re-number type id's 
    sort!(new_types, by = x -> x.id)
    for i in 1:length(new_types)
      type = new_types[i]
      if type.id != i
        foreach(o -> o.type.id = i, filter(obj -> obj.type.id == type.id, new_objects))
        type.id = i
      end
    end

    # re-number object id's
    sort!(new_objects, by = x -> x.id)
    for i in 1:length(new_objects)
      object = new_objects[i]
      object.id = i
    end

    for type in new_types 
      if (type.custom_fields != []) && (length(type.custom_fields[1][3]) == 1)
        type.color = type.custom_fields[1][3][1]
        type.custom_fields = []
      end
    end 

    for object in new_objects 
      if object.type.custom_fields == []
        object.custom_field_values = []
      end
    end
  end

  # println("BEFORE COMBINING TYPES INTO ONE WITH COLOR FIELD")
  # # @show new_types 
  # # @show new_objects 

  # group objects with same shape but different colors into one type
  new_types, new_objects = combine_types_with_same_shape(new_types, new_objects)
  # println("POST COMBINING")
  # # @show new_types 
  # # @show new_objects 

  # take the union of override_types and new_types
  # # @show old_override_types
  old_types = deepcopy(old_override_types)
  new_object_colors = map(o -> o.type.color, new_objects)
  types_to_add = []
  for new_type in new_types
    new_type_shape = new_type.shape
    if new_type_shape in map(t -> t.shape, old_types)
      override_type = old_types[findall(t -> t.shape == new_type_shape, old_types)[1]]
      if length(override_type.custom_fields) == 0
        if length(new_type.custom_fields) == 0
          if override_type.color != new_type.color 
            push!(override_type.custom_fields, ("color", "String", [override_type.color, new_type.color]))
          end
        else
          push!(override_type.custom_fields, ("color", "String", unique([override_type.color, new_type.custom_fields[1][3]...])))
        end
      else
        if length(new_type.custom_fields) == 0
          colors = override_type.custom_fields[1][3]
          push!(colors, new_type.color)
          unique!(colors)
        else
          push!(override_type.custom_fields[1][3], new_type.custom_fields[1][3]...)
          unique!(override_type.custom_fields[1][3])
        end
      end
    else
      new_type.id = length(old_types) + length(types_to_add) + 1
      push!(types_to_add, new_type)
    end
  end 
  new_types = vcat(old_types..., types_to_add...)

  for i in 1:length(new_types) 
    new_types[i].id = i
  end

  # println("POST UNION")
  # println(new_types)

  # reassign objects 
  for i in 1:length(new_objects)
    object = new_objects[i]
    # new type
    type = new_types[findall(t -> t.shape == object.type.shape, new_types)[1]]
    if type.custom_fields != [] && object.custom_field_values == []
      push!(object.custom_field_values, new_object_colors[i])
    end
    object.type = type 
  end

  (new_types, new_objects, background, dim)

end

function parsescene_autumn_given_types_2x2(render_output::AbstractArray, square_colors::AbstractArray, dim::Int=16, background::String="white")
  render_output_2x2_colors = sort(filter(cell -> cell.color in square_colors, render_output), by=c -> (c.position.x, c.position.y))
  render_output_other = filter(cell -> !(cell.color in square_colors) , render_output)

  types_other, objects_other, _, _ = parsescene_autumn(render_output_other, dim, background)

  types_2x2 = [] 
  objects_2x2 = []

  push!(types_2x2, ObjType([(0, -1), (0, 0), (1, -1), (1, 0)], 
                           square_colors[1], 
                           length(square_colors) > 1 ? [("color", "String", square_colors)] : [], 
                           length(types_other) + 1))

  while !isempty(render_output_2x2_colors)
    cell = render_output_2x2_colors[1]
    remaining_cells = unique(filter(c -> c.color == cell.color && 
                                  ((c.position.x, c.position.y) in [(cell.position.x + 1, cell.position.y),
                                                                    (cell.position.x, cell.position.y + 1),
                                                                    (cell.position.x + 1, cell.position.y + 1)])   
                             , render_output_2x2_colors))
    if length(remaining_cells) != 3 
      return ([], [], background, dim)
    end

    push!(objects_2x2, Obj(types_2x2[1], 
                           (cell.position.x, cell.position.y + 1), 
                           length(square_colors) > 1 ? [cell.color] : [], 
                           length(objects_other) + length(objects_2x2) + 1))

    for c in [cell, remaining_cells...]
      index = findall(x -> x == c, render_output_2x2_colors)[1]
      deleteat!(render_output_2x2_colors, index)
    end
    # filter!(c -> !(c in [cell, remaining_cells...]), remaining_cells)
  end

  new_types = [types_other..., types_2x2...]
  new_objects = [objects_other..., objects_2x2...]
  
  (new_types, new_objects, background, dim)
end

function combine_types_with_same_shape(object_types, objects)
  # println("COMBINE TYPES WITH SAME SHAPE")
  # println(object_types)
  # println(objects)
  types_to_remove = []
  for i in 1:length(object_types)
    type_i = object_types[i]
    type_i_shape = type_i.shape
    for j in i:length(object_types)
      type_j = object_types[j] 
      type_j_shape = object_types[j].shape 
      if (i != j) && (type_i_shape == type_j_shape)
        push!(types_to_remove, type_j)
        if "color" in map(tuple -> tuple[1], type_i.custom_fields)
          colors = type_i.custom_fields[findall(tuple -> tuple[1] == "color", type_i.custom_fields)[1]][3]
          push!(colors, type_j.color)
          unique!(colors)
        else
          push!(type_i.custom_fields, ("color", "String", [type_i.color, type_j.color]))
          unique!(type_i.custom_fields)

          objects_to_update_type_i = filter(obj -> obj.type.id == type_i.id, objects)
          foreach(o -> push!(o.custom_field_values, o.type.color) , objects_to_update_type_i)
        end
        # objects_to_update_type_i = filter(obj -> obj.type.id == type_i.id, objects)
        # foreach(o -> push!(o.custom_field_values, o.type.color) , objects_to_update_type_i)

        objects_to_update_type_j = filter(obj -> obj.type.id == type_j.id, objects)
        foreach(o -> o.type = type_i, objects_to_update_type_j)
        foreach(o -> o.custom_field_values = o.custom_field_values == [] ? [type_j.color] : o.custom_field_values, objects_to_update_type_j)
      end
    end
  end
  object_types = filter(type -> !(type in types_to_remove), object_types)

  # re-number type id's 
  sort!(object_types, by = x -> x.id)
  for i in 1:length(object_types)
    type = object_types[i]
    if type.id != i
      foreach(o -> o.type.id = i, filter(obj -> obj.type.id == type.id, objects))
      type.id = i
    end
  end

  # re-number object id's
  sort!(objects, by = x -> x.id)
  for i in 1:length(objects)
    object = objects[i]
    object.id = i
  end


  # println("END COMBINE TYPES WITH SAME SHAPE")
  # println(object_types)
  # println(objects)
  (object_types, objects)
end

"""
mutable struct ObjType
  shape::AbstractArray
  color::String
  custom_fields::AbstractArray
  id::Int
end

mutable struct Obj
  type::ObjType
  position::Tuple{Int, Int}
  custom_field_values::AbstractArray
  id::Int
end
"""

function parsescene_autumn_singlecell(render_output::AbstractArray, background::String="white", dim::Int=16)
  colors = unique(map(cell -> cell.color, render_output))
  types = map(color -> ObjType([(0,0)], color, [], findall(c -> c == color, colors)[1]), colors)
  objects = []
  for i in 1:length(render_output)
    cell = render_output[i]
    push!(objects, Obj(types[findall(type -> type.color == cell.color, types)[1]], (cell.position.x, cell.position.y), [], i))
  end
  (types, objects, background, dim)
end

function parsescene_autumn_singlecell_given_types(render_output::AbstractArray, override_types::AbstractArray, background::String="white", dim::Int=16)
  standard_types, objects, _, _ = parsescene_autumn_singlecell(render_output, background, dim)
  # println("STANDARD TYPES ")
  # println(standard_types)
  standard_types = deepcopy(standard_types)
  override_types = deepcopy(override_types)
  # compute union of standard types and override_types 
  new_types = filter(type -> !(type.color in map(t -> t.color, override_types)), standard_types)
  for i in 1:length(new_types) 
    type = new_types[i]
    type.id = length(override_types) + i
  end
  # println("RETURN VAL")
  # println(vcat(override_types..., new_types...))
  final_types = vcat(override_types..., new_types...)
  # ensure objects have correct types 
  for object in objects
    object.type = final_types[findall(t -> t.color == object.type.color, final_types)[1]]
  end

  (final_types, objects, background, dim)
end

# ----- end functions related to scene parsing ----- #

# ----- PEDRO functions ----- # 

function parsescene_autumn_pedro(render_output, gridsize=16, background::String="white")
  dim = gridsize[1]
  objectshapes = []
  
  render_output_copy = deepcopy(render_output)
  while !isempty(render_output_copy)
    min_x = minimum(map(cell -> cell.position.x, render_output_copy))
    min_y = minimum(map(cell -> cell.position.y, filter(cell -> cell.position.x == min_x, render_output_copy)))
    color = filter(cell -> cell.position.x == min_x && cell.position.y == min_y, render_output_copy)[1].color

    shape = unique(filter(cell -> (cell.position.x in collect(min_x:min_x + 29)) && (cell.position.y in collect(min_y:min_y + 29)) && cell.color == color, render_output_copy))
    # remove shape from render_output_copy 
    for cell in shape 
      matching_cell_indices = findall(c -> c == cell, render_output_copy)
      deleteat!(render_output_copy, matching_cell_indices[1])
    end

    shape = map(cell -> ((cell.position.x, cell.position.y), cell.color), shape)
    push!(objectshapes, shape)    
  end
  # # # @show objectshapes

  types = []
  objects = []
  # # # @show length(objectshapes)
  for objectshape_with_color in objectshapes
    objectcolor = objectshape_with_color[1][2]
    objectshape = map(o -> o[1], objectshape_with_color)
    # # # @show objectcolor 
    # # # @show objectshape
    translated = map(pos -> dim * pos[2] + pos[1], objectshape)
    translated = length(translated) % 2 == 0 ? translated[1:end-1] : translated # to produce a single median
    centerPos = objectshape[findall(x -> x == median(translated), translated)[1]]
    translatedShape = unique(map(pos -> (pos[1] - centerPos[1], pos[2] - centerPos[2]), sort(objectshape)))

    if !(objectcolor in map(type -> type.color , types))
      push!(types, ObjType(translatedShape, objectcolor, [], length(types) + 1))
      push!(objects, Obj(types[length(types)], centerPos, [], length(objects) + 1))
    else
      type_id = findall(type -> type.color == objectcolor, types)[1]
      push!(objects, Obj(types[type_id], centerPos, [], length(objects) + 1))
    end
  end

  # combine types with the same shape but different colors
  # println("INSIDE REGULAR PARSER")
  # # # @show types 
  # # # @show objects
  # (types, objects) = combine_types_with_same_shape(types, objects)

  (types, objects, background, dim)
end

function parsescene_autumn_pedro_given_types(render_output, types, gridsize=16, background::String="white")
  (_, objects, _, _) = parsescene_autumn_pedro(render_output, gridsize, background)

  new_objects = []
  # reassign types to objects 
  for object in objects 
    type = filter(t -> t.color == object.type.color, types)[1]
    push!(new_objects, Obj(type, object.position, [], object.id))
  end
  (types, new_objects, background, gridsize) 
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

struct Object 
  id::Int 
  origin::Position
  type::Symbol
  alive::Bool 
  changed::Bool
  custom_fields::Dict{Symbol, Union{Int, String, Bool}}
  render::Union{Nothing, AbstractArray}
end

struct ObjectType
  render::Union{Nothing, AExpr, AbstractArray}
  fields::Array{AExpr}
end

mutable struct Scene 
  objects::Array{Object}
  background::String
end

mutable struct State 
  time::Int 
  objectsCreated::Int 
  rng::AbstractRNG
  scene::Scene 
  object_types::Dict{Symbol, ObjectType}
  histories::Dict{Symbol, Dict{Int, Union{Int, String, Bool, Position, Object, AbstractArray}}}
end


mutable struct Env 
  left::Bool 
  right::Bool 
  up::Bool 
  down::Bool
  click::Union{Nothing, Click}
  current_var_values::Dict{Symbol, Union{Object, Int, Bool, String, Position, State, AbstractArray}}
  lifted::Dict{Symbol, Union{AExpr, BigInt, Int, String}}
  on_clauses::Dict{Symbol, Array{Union{AExpr, Symbol}}}
  state::State
end

function update_nt(object::Object, x::Symbol, v)
  if x == :id 
    object = @set object.id = v
  elseif x == :origin 
    object = @set object.origin = v
  elseif x == :type 
    object = @set object.type = v
  elseif x == :alive 
    object = @set object.alive = v
  elseif x == :changed 
    object = @set object.changed = v
  elseif x == :custom_fields 
    object = @set object.custom_fields = v
  elseif x == :render
    object = @set object.render = v
  else
    object = deepcopy(object)
    object.custom_fields[x] = v
  end
  object
end

Click(x, y, @nospecialize(state::State)) = Click(x, y)
Position(x, y, @nospecialize(state::State)) = Position(x, y) 

Cell(position::Position, color::String) = Cell(position, color, 0.8)
Cell(x::Int, y::Int, color::String) = Cell(Position(floor(Int, x), floor(Int, y)), color, 0.8)
# Cell(x::Int, y::Int, color::String, opacity::Float64) = Cell(Position(floor(Int, x), floor(Int, y)), color, opacity)

Cell(x, y, color::String, @nospecialize(state::State)) = Cell(floor(Int, x), floor(Int, y), color)
Cell(position::Position, color::String, @nospecialize(state::State)) = Cell(position::Position, color::String)

# struct Scene
#   objects::Array{Object}
#   background::String
# end

# Scene(@nospecialize(objects::AbstractArray)) = Scene(objects, "#ffffff00")

# function render(scene)::Array{Cell}
#   vcat(map(obj -> render(obj), filter(obj -> obj.alive, scene.objects))...)
# end

function prev(obj::Object, @nospecialize(state))
  prev_objects = filter(o -> o.id == obj.id, state.scene.objects)
  if prev_objects != []
    prev_objects[1]                            
  else
    obj
  end
end

function render(obj::Object, state::Union{State, Nothing}=nothing)::Array{Cell}
  if obj.alive
    if isnothing(obj.render)
      render = state.object_types[obj.type].render
      map(cell -> Cell(move(cell.position, obj.origin), cell.color), render)
    else
      map(cell -> Cell(move(cell.position, obj.origin), cell.color), obj.render)
    end
  else
    []
  end
end

function renderScene(@nospecialize(scene::Scene), state::Union{State, Nothing}=nothing)
  vcat(map(o -> render(o, state), filter(x -> x.alive, scene.objects))...)
end

function occurred(click, state::Union{State, Nothing}=nothing)
  !isnothing(click)
end

function uniformChoice(freePositions, @nospecialize(state::State))
  freePositions[rand(state.rng, Categorical(ones(length(freePositions))/length(freePositions)))]
end

function uniformChoice(freePositions, n::Union{Int, BigInt}, @nospecialize(state::State))
  map(idx -> freePositions[idx], rand(state.rng, Categorical(ones(length(freePositions))/length(freePositions)), n))
end

function min(arr, state::Union{State, Nothing}=nothing)
  Base.min(arr...)
end

function range(start::Int, stop::Int, state::Union{State, Nothing}=nothing)
  [start:stop;]
end

function isWithinBounds(obj::Object, @nospecialize(state::State))::Bool
  # # println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
  length(filter(cell -> !isWithinBounds(cell.position, state), render(obj, state))) == 0
end

function isOutsideBounds(obj::Object, @nospecialize(state::State))::Bool
  # # println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
  length(filter(cell -> isWithinBounds(cell.position, state), render(obj, state))) == 0
end

function clicked(click::Union{Click, Nothing}, object::Object, @nospecialize(state::State))::Bool
  if isnothing(click)
    false
  else
    GRID_SIZE = state.histories[:GRID_SIZE][0]
    if GRID_SIZE isa AbstractArray 
      GRID_SIZE_X = GRID_SIZE[1]
      nums = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(object, state))
      (GRID_SIZE_X * click.y + click.x) in nums
    else
      nums = map(cell -> GRID_SIZE*cell.position.y + cell.position.x, render(object, state))
      (GRID_SIZE * click.y + click.x) in nums
    end

  end
end

function clicked(click::Union{Click, Nothing}, @nospecialize(objects::AbstractArray), @nospecialize(state::State))  
  # # println("LOOK AT ME")
  # # println(reduce(&, map(obj -> clicked(click, obj), objects)))
  if isnothing(click)
    false
  else
    foldl(|, map(obj -> clicked(click, obj, state), objects), init=false)
  end
end

function objClicked(click::Union{Click, Nothing}, @nospecialize(objects::AbstractArray), state::Union{State, Nothing}=nothing)::Union{Object, Nothing}
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

function clicked(click::Union{Click, Nothing}, x::Int, y::Int, @nospecialize(state::State))::Bool
  if click == nothing
    false
  else
    click.x == x && click.y == y                         
  end
end

function clicked(click::Union{Click, Nothing}, pos::Position, @nospecialize(state::State))::Bool
  if click == nothing
    false
  else
    click.x == pos.x && click.y == pos.y                         
  end
end

function pushConfiguration(arrow::Position, @nospecialize(obj1::Object), @nospecialize(obj2::Object), @nospecialize(state::State))
  pushConfiguration(arrow, obj1, [obj2], state)
end

function pushConfiguration(arrow::Position, @nospecialize(obj1::Object), @nospecialize(obj2::AbstractArray), @nospecialize(state::State))
  println("pushConfiguration: obj1 id = $(obj1.id), time = $(state.time)")
  moveIntersects(arrow, obj1, obj2, state) && isFree(move(move(obj1, arrow, state), arrow, state).origin, state)
end

function pushConfiguration(arrow::Position, @nospecialize(obj1::Object), @nospecialize(obj2::AbstractArray), @nospecialize(obj3::AbstractArray), @nospecialize(state::State))
  moveIntersects(arrow, obj1, obj2, state) && intersects(move(move(obj1, arrow, state), arrow, state), obj3, state)
end

function moveIntersects(arrow::Position, @nospecialize(obj1::Object), @nospecialize(obj2::Object), @nospecialize(state::State)) 
  (arrow != Position(0, 0)) && intersects(move(obj1, arrow, state), obj2, state)
end

function moveIntersects(arrow::Position, @nospecialize(obj::Object), @nospecialize(objects::AbstractArray), @nospecialize(state::State)) 
  (arrow != Position(0, 0)) && intersects(move(obj, arrow, state), objects, state)
end

function intersects(@nospecialize(obj1::Object), @nospecialize(obj2::Object), @nospecialize(state::State))::Bool
  GRID_SIZE = state.histories[:GRID_SIZE][0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj2, state))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, render(obj2, state))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj1::Object), @nospecialize(obj2::AbstractArray), @nospecialize(state::State))::Bool
  GRID_SIZE = state.histories[:GRID_SIZE][0]
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj2::AbstractArray), @nospecialize(obj1::Object), @nospecialize(state::State))::Bool
  GRID_SIZE = state.histories[:GRID_SIZE][0] 
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
    length(intersect(nums1, nums2)) != 0
  else
    nums1 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, render(obj1, state))
    nums2 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
    length(intersect(nums1, nums2)) != 0
  end
end

function intersects(@nospecialize(obj1::AbstractArray), @nospecialize(obj2::AbstractArray), @nospecialize(state::State))::Bool
  if (length(obj1) == 0) || (length(obj2) == 0)
    false  
  elseif (obj1[1] isa Object) && (obj2[1] isa Object)
    # # println("MADE IT")
    GRID_SIZE = state.histories[:GRID_SIZE][0]
    if GRID_SIZE isa AbstractArray 
      GRID_SIZE_X = GRID_SIZE[1]
      nums1 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj1)...))
      nums2 = map(cell -> GRID_SIZE_X*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
      length(intersect(nums1, nums2)) != 0
    else
      nums1 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj1)...))
      nums2 = map(cell -> state.histories[:GRID_SIZE][0]*cell.position.y + cell.position.x, vcat(map(o -> render(o, state), obj2)...))
      length(intersect(nums1, nums2)) != 0
    end
  else
    length(intersect(obj1, obj2)) != 0 
  end
end

function intersects(object::Object, @nospecialize(state::State))::Bool
  objects = state.scene.objects
  intersects(object, objects, state)
end

function addObj(@nospecialize(list::AbstractArray), obj::Object, state::Union{State, Nothing}=nothing)
  obj = update_nt(obj, :changed, true)
  new_list = vcat(list, obj)
  new_list
end

function addObj(@nospecialize(list::AbstractArray), @nospecialize(objs::AbstractArray), state::Union{State, Nothing}=nothing)
  objs = map(obj -> update_nt(obj, :changed, true), objs)
  new_list = vcat(list, objs)
  new_list
end

function removeObj(@nospecialize(list::AbstractArray), obj::Object, state::Union{State, Nothing}=nothing)
  new_list = deepcopy(list)
  for x in filter(o -> o.id == obj.id, new_list)
    index = findall(o -> o.id == x.id, new_list)[1]
    new_list[index] = update_nt(update_nt(x, :alive, false), :changed, true)
    #x.alive = false 
    #x.changed = true
  end
  new_list
end

function removeObj(@nospecialize(list::AbstractArray), fn, state::Union{State, Nothing}=nothing)
  new_list = deepcopy(list)
  for x in filter(obj -> fn(obj), new_list)
    index = findall(o -> o.id == x.id, new_list)[1]
    new_list[index] = update_nt(update_nt(x, :alive, false), :changed, true)
    #x.alive = false 
    #x.changed = true
  end
  new_list
end

function removeObj(obj::Object, state::Union{State, Nothing}=nothing)
  new_obj = deepcopy(obj)
  new_obj = update_nt(update_nt(new_obj, :alive, false), :changed, true)
  # new_obj.alive = false
  # new_obj.changed = true
  # new_obj
end

function updateObj(obj::Object, field::String, value, state::Union{State, Nothing}=nothing)
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

function filter_fallback(obj::Object, state::Union{State, Nothing}=nothing)
  true
end

# function updateObj(@nospecialize(list::AbstractArray), map_fn, filter_fn, state::Union{State, Nothing}=nothing)
#   orig_list = filter(obj -> !filter_fn(obj), list)
#   filtered_list = filter(filter_fn, list)
#   new_filtered_list = map(map_fn, filtered_list)
#   foreach(obj -> obj.changed = true, new_filtered_list)
#   vcat(orig_list, new_filtered_list)
# end

# function updateObj(@nospecialize(list::AbstractArray), map_fn, state::Union{State, Nothing}=nothing)
#   orig_list = filter(obj -> false, list)
#   filtered_list = filter(obj -> true, list)
#   new_filtered_list = map(map_fn, filtered_list)
#   foreach(obj -> obj.changed = true, new_filtered_list)
#   vcat(orig_list, new_filtered_list)
# end

function adjPositions(position::Position, @nospecialize(state::State))::Array{Position}
  filter(x -> isWithinBounds(x, state), [Position(position.x, position.y + 1), Position(position.x, position.y - 1), Position(position.x + 1, position.y), Position(position.x - 1, position.y)])
end

function isWithinBounds(position::Position, @nospecialize(state::State))::Bool
  GRID_SIZE = state.histories[:GRID_SIZE][0] 
  if GRID_SIZE isa AbstractArray 
    GRID_SIZE_X = GRID_SIZE[1]
    GRID_SIZE_Y = GRID_SIZE[2]
  else
    GRID_SIZE_X = GRID_SIZE
    GRID_SIZE_Y = GRID_SIZE
  end
  (position.x >= 0) && (position.x < GRID_SIZE_X) && (position.y >= 0) && (position.y < GRID_SIZE_Y)                  
end

function isOutsideBounds(position::Position, @nospecialize(state::State))::Bool
  !isWithinBounds(position, state)
end

function isFree(position::Position, @nospecialize(state::State))::Bool
  length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, renderScene(state.scene, state))) == 0
end

function isFree(click::Union{Click, Nothing}, @nospecialize(state::State))::Bool
  if click == nothing
    false
  else
    isFree(Position(click.x, click.y), state)
  end
end

function isFree(positions::Array{Position}, @nospecialize(state::State))::Bool 
  foldl(|, map(pos -> isFree(pos, state), positions), init=false)
end

function rect(pos1::Position, pos2::Position, state::Union{State, Nothing}=nothing)
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

function unitVector(position1::Position, position2::Position, @nospecialize(state::State))::Position
  deltaX = position2.x - position1.x
  deltaY = position2.y - position1.y
  if (floor(Int, abs(sign(deltaX))) == 1 && floor(Int, abs(sign(deltaY))) == 1)
    Position(sign(deltaX), 0)
    # uniformChoice(rng, [Position(sign(deltaX), 0), Position(0, sign(deltaY))])
  else
    Position(sign(deltaX), sign(deltaY))  
  end
end

function unitVector(object1::Object, object2::Object, @nospecialize(state::State))::Position
  position1 = object1.origin
  position2 = object2.origin
  unitVector(position1, position2, state)
end

function unitVector(object::Object, position::Position, @nospecialize(state::State))::Position
  unitVector(object.origin, position, state)
end

function unitVector(position::Position, object::Object, @nospecialize(state::State))::Position
  unitVector(position, object.origin, state)
end

function unitVector(position::Position, @nospecialize(state::State))::Position
  unitVector(Position(0,0), position, state)
end 

function displacement(position1::Position, position2::Position, state::Union{State, Nothing}=nothing)::Position
  Position(floor(Int, position2.x - position1.x), floor(Int, position2.y - position1.y))
end

function displacement(cell1::Cell, cell2::Cell, state::Union{State, Nothing}=nothing)::Position
  displacement(cell1.position, cell2.position)
end

function adjacent(position1::Position, position2::Position, unitSize::Int, state::Union{State, Nothing}=nothing)::Bool
  displacement(position1, position2) in [Position(0, unitSize), Position(unitSize, 0), Position(0, -unitSize), Position(-unitSize, 0)]
end

function adjacent(cell1::Cell, cell2::Cell, unitSize::Int, state::Union{State, Nothing}=nothing)::Bool
  adjacent(cell1.position, cell2.position, unitSize)
end

function adjacent(cell::Cell, cells::Array{Cell}, unitSize::Int, state::Union{State, Nothing}=nothing)
  length(filter(x -> adjacent(cell, x, unitSize), cells)) != 0
end

function adjacentObjs(obj::Object, unitSize::Int, @nospecialize(state::State))
  filter(o -> adjacent(o.origin, obj.origin, unitSize) && (obj.id != o.id), state.scene.objects)
end

function adjacentObjsDiag(obj::Object, @nospecialize(state::State))
  filter(o -> adjacentDiag(o.origin, obj.origin) && (obj.id != o.id), state.scene.objects)
end

function adjacentDiag(position1::Position, position2::Position, state::Union{State, Nothing}=nothing)
  displacement(position1, position2) in [Position(0,1), Position(1, 0), Position(0, -1), Position(-1, 0),
                                         Position(1,1), Position(1, -1), Position(-1, 1), Position(-1, -1)]
end

function adj(@nospecialize(obj1::Object), @nospecialize(obj2::Object), unitSize::Int, @nospecialize(state::State)) 
  filter(o -> o.id == obj2.id, adjacentObjs(obj1, unitSize, state)) != []
end

function adj(@nospecialize(obj1::Object), @nospecialize(obj2::AbstractArray), unitSize::Int, @nospecialize(state::State)) 
  filter(o -> o.id in map(x -> x.id, obj2), adjacentObjs(obj1, unitSize, state)) != []
end

function adj(@nospecialize(obj1::AbstractArray), @nospecialize(obj2::AbstractArray), unitSize::Int, @nospecialize(state::State)) 
  obj1_adjacentObjs = vcat(map(x -> adjacentObjs(x, unitSize, state), obj1)...)
  intersect(map(x -> x.id, obj1_adjacentObjs), map(x -> x.id, obj2)) != []  
end

function rotate(object::Object, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :render, map(x -> Cell(rotate(x.position), x.color), new_object.render))
  new_object
end

function rotate(position::Position, state::Union{State, Nothing}=nothing)::Position
  Position(-position.y, position.x)
 end

function rotateNoCollision(object::Object, @nospecialize(state::State))::Object
  (isWithinBounds(rotate(object), state) && isFree(rotate(object), object, state)) ? rotate(object) : object
end

function move(position1::Position, position2::Position, state::Union{State, Nothing}=nothing)
  Position(position1.x + position2.x, position1.y + position2.y)
end

function move(position::Position, cell::Cell, state::Union{State, Nothing}=nothing)
  Position(position.x + cell.position.x, position.y + cell.position.y)
end

function move(cell::Cell, position::Position, state::Union{State, Nothing}=nothing)
  Position(position.x + cell.position.x, position.y + cell.position.y)
end

function move(object::Object, position::Position, state::Union{State, Nothing}=nothing)
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, move(object.origin, position))
  new_object
end

function move(object::Object, x::Int, y::Int, state::Union{State, Nothing}=nothing)::Object
  move(object, Position(x, y))                          
end

# ----- begin left/right move ----- #

function moveLeft(object::Object, state::Union{State, Nothing}=nothing)::Object
  move(object, Position(-1, 0))                          
end

function moveRight(object::Object, state::Union{State, Nothing}=nothing)::Object
  move(object, Position(1, 0))                          
end

function moveUp(object::Object, state::Union{State, Nothing}=nothing)::Object
  move(object, Position(0, -1))                          
end

function moveDown(object::Object, state::Union{State, Nothing}=nothing)::Object
  move(object, Position(0, 1))                          
end

# ----- end left/right move ----- #

function moveNoCollision(object::Object, position::Position, @nospecialize(state::State))::Object
  (isWithinBounds(move(object, position), state) && isFree(move(object, position.x, position.y), object, state)) ? move(object, position.x, position.y) : object 
end

function moveNoCollision(object::Object, x::Int, y::Int, @nospecialize(state::State))
  (isWithinBounds(move(object, x, y), state) && isFree(move(object, x, y), object, state)) ? move(object, x, y) : object 
end

function moveNoCollisionColor(object::Object, position::Position, color::String, @nospecialize(state::State))::Object
  new_object = move(object, position) 
  matching_color_objects = filter(obj -> intersects(new_object, obj, state) && (color in map(cell -> cell.color, render(obj, state))), state.scene.objects)
  if matching_color_objects == []
    new_object
  else
    object 
  end
end

function moveNoCollisionColor(object::Object, x::Int, y::Int, color::String, @nospecialize(state::State))::Object
  new_object = move(object, Position(x, y)) 
  matching_color_objects = filter(obj -> intersects(new_object, obj, state) && (color in map(cell -> cell.color, render(obj, state))), state.scene.objects)
  if matching_color_objects == []
    new_object
  else
    object 
  end
end

# ----- begin left/right moveNoCollision ----- #

function moveLeftNoCollision(object::Object, @nospecialize(state::State))::Object
  (isWithinBounds(move(object, -1, 0), state) && isFree(move(object, -1, 0), object, state)) ? move(object, -1, 0) : object 
end

function moveRightNoCollision(object::Object, @nospecialize(state::State))::Object
  x = (isWithinBounds(move(object, 1, 0), state) && isFree(move(object, 1, 0), object, state)) ? move(object, 1, 0) : object 
  x
end

function moveUpNoCollision(object::Object, @nospecialize(state::State))::Object
  (isWithinBounds(move(object, 0, -1), state) && isFree(move(object, 0, -1), object, state)) ? move(object, 0, -1) : object 
end

function moveDownNoCollision(object::Object, @nospecialize(state::State))::Object
  (isWithinBounds(move(object, 0, 1), state) && isFree(move(object, 0, 1), object, state)) ? move(object, 0, 1) : object 
end

# ----- end left/right moveNoCollision ----- #

function moveWrap(object::Object, position::Position, @nospecialize(state::State))::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, position.x, position.y, state))
  new_object
end

function moveWrap(cell::Cell, position::Position, @nospecialize(state::State))
  moveWrap(cell.position, position.x, position.y, state)
end

function moveWrap(position::Position, cell::Cell, @nospecialize(state::State))
  moveWrap(cell.position, position, state)
end

function moveWrap(object::Object, x::Int, y::Int, @nospecialize(state::State))::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, x, y, state))
  new_object
end

function moveWrap(position1::Position, position2::Position, @nospecialize(state::State))::Position
  moveWrap(position1, position2.x, position2.y, state)
end

function moveWrap(position::Position, x::Int, y::Int, @nospecialize(state::State))::Position
  GRID_SIZE = state.histories[:GRID_SIZE][0]
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

function moveLeftWrap(object::Object, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, -1, 0, state))
  new_object
end
  
function moveRightWrap(object::Object, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 1, 0, state))
  new_object
end

function moveUpWrap(object::Object, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 0, -1, state))
  new_object
end

function moveDownWrap(object::Object, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, moveWrap(object.origin, 0, 1, state))
  new_object
end

# ----- end left/right moveWrap ----- #

function randomPositions(GRID_SIZE, n::Int, state::Union{State, Nothing}=nothing)::Array{Position}
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

function distance(position1::Position, position2::Position, state::Union{State, Nothing}=nothing)::Int
  abs(position1.x - position2.x) + abs(position1.y - position2.y)
end

function distance(object1::Object, object2::Object, state::Union{State, Nothing}=nothing)::Int
  position1 = object1.origin
  position2 = object2.origin
  distance(position1, position2)
end

function distance(object::Object, position::Position, state::Union{State, Nothing}=nothing)::Int
  distance(object.origin, position)
end

function distance(position::Position, object::Object, state::Union{State, Nothing}=nothing)::Int
  distance(object.origin, position)
end

function distance(object::Object, @nospecialize(objects::AbstractArray), state::Union{State, Nothing}=nothing)::Int
  if objects == []
    typemax(Int)
  else
    distances = map(obj -> distance(object, obj), objects)
    minimum(distances)
  end
end

function distance(@nospecialize(objects1::AbstractArray), @nospecialize(objects2::AbstractArray), state::Union{State, Nothing}=nothing)::Int
  if objects1 == [] || objects2 == []
    typemax(Int)
  else
    distances = vcat(map(obj -> distance(obj, objects2), objects1)...)
    minimum(distances)
  end
end


function firstWithDefault(@nospecialize(arr::AbstractArray), state::Union{State, Nothing}=nothing) 
  if arr == [] 
    Position(-30, -30)
  else 
    first(arr)
  end
end

function farthestRandom(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  choices = [farthestLeft(object, types, unit_size, state), 
             farthestRight(object, types, unit_size, state), 
             farthestDown(object, types, unit_size, state), 
             farthestUp(object, types, unit_size, state)]

  nonzero_positions = filter(p -> p != Position(0, 0), choices)

  # println("farthestRandom")
  # @show nonzero_positions 

  if nonzero_positions == [] 
    Position(0, 0)
  else
    rand(nonzero_positions)
  end
end

function farthestLeft(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position 
  orig_position = closestRight(object, types, unit_size, state)
  if orig_position == Position(unit_size, 0)
    Position(-unit_size, 0)
  else
    objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
    if length(objects_of_type) == 0 
      Position(0, 0)
    else
      min_distance = min(map(obj -> distance(object, obj), objects_of_type))
      objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
      if objects_of_min_distance[1].origin.x == object.origin.x
        Position(-unit_size, 0)
      else
        Position(0, 0)
      end
    end
  end
end

function farthestRight(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  orig_position = closestLeft(object, types, unit_size, state)
  if orig_position == Position(-unit_size, 0) 
    Position(unit_size, 0)
  else
    objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
    if length(objects_of_type) == 0 
      Position(0, 0)
    else
      min_distance = min(map(obj -> distance(object, obj), objects_of_type))
      objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
      if objects_of_min_distance[1].origin.x == object.origin.x
        Position(unit_size, 0)
      else
        Position(0, 0)
      end
    end
  end
end

function farthestUp(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  orig_position = closestDown(object, types, unit_size, state)
  if orig_position == Position(0, unit_size) 
    Position(0, -unit_size)
  else
    objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
    if length(objects_of_type) == 0 
      Position(0, 0)
    else
      min_distance = min(map(obj -> distance(object, obj), objects_of_type))
      objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
      if objects_of_min_distance[1].origin.y == object.origin.y
        Position(0, -unit_size)
      else
        Position(0, 0)
      end
    end
  end
end

function farthestDown(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  orig_position = closestUp(object, types, unit_size, state)
  if orig_position == Position(0, -unit_size) 
    Position(0, unit_size)
  else
    objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
    if length(objects_of_type) == 0 
      Position(0, 0)
    else
      min_distance = min(map(obj -> distance(object, obj), objects_of_type))
      objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
      if objects_of_min_distance[1].origin.y == object.origin.y
        Position(0, unit_size)
      else
        Position(0, 0)
      end
    end
  end
end

function closest(object::Object, type::Symbol, @nospecialize(state::State))::Position
  objects_of_type = filter(obj -> (obj.type == type) && (obj.alive), state.scene.objects)
  if length(objects_of_type) == 0
    object.origin
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    sort(objects_of_min_distance, by=o -> (o.origin.x, o.origin.y))[1].origin
  end
end

function closest(object::Object, @nospecialize(types::AbstractArray), @nospecialize(state::State))::Position
  objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
  if length(objects_of_type) == 0
    object.origin
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    sort(objects_of_min_distance, by=o -> (o.origin.x, o.origin.y))[1].origin
  end
end

function closestRandom(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  choices = [closestLeft(object, types, unit_size, state), 
             closestRight(object, types, unit_size, state), 
             closestDown(object, types, unit_size, state), 
             closestUp(object, types, unit_size, state)]

  nonzero_positions = filter(p -> p != Position(0, 0), choices)

  # println("closestRandom")
  # @show nonzero_positions 

  if nonzero_positions == [] 
    Position(0, 0)
  else
    rand(nonzero_positions)
  end
end

function closestLeft(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
  if length(objects_of_type) == 0
    Position(0, 0)
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    negative_x_displacements = filter(x -> x < 0, map(o -> (o.origin.x - object.origin.x), objects_of_min_distance))
    if length(negative_x_displacements) > 0
      Position(-unit_size, 0)
    else
      Position(0, 0)        
    end
  end
end

function closestRight(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
  if length(objects_of_type) == 0
    Position(0, 0)
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    positive_x_displacements = filter(x -> x > 0, map(o -> (o.origin.x - object.origin.x), objects_of_min_distance))
    if length(positive_x_displacements) > 0
      Position(unit_size, 0)
    else
      Position(0, 0)        
    end
  end
end

function closestUp(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  # @show object 
  # @show types 
  # @show state   
  
  objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
  # @show objects_of_type 
  if length(objects_of_type) == 0
    Position(0, 0)
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    negative_y_displacements = filter(x -> x < 0, map(o -> (o.origin.y - object.origin.y), objects_of_min_distance))

    # @show min_distance 
    # @show objects_of_min_distance 
    # @show negative_y_displacements 

    if length(negative_y_displacements) > 0
      Position(0, -unit_size)
    else
      Position(0, 0)
    end
  end
end

function closestDown(object::Object, @nospecialize(types::AbstractArray), unit_size::Int, @nospecialize(state::State))::Position
  # @show object 
  # @show types 
  # @show state 
  objects_of_type = filter(obj -> (obj.type in types) && (obj.alive), state.scene.objects)
  # @show objects_of_type 
  if length(objects_of_type) == 0
    Position(0, 0)
  else
    min_distance = min(map(obj -> distance(object, obj), objects_of_type))
    objects_of_min_distance = filter(obj -> distance(object, obj) == min_distance, objects_of_type)
    positive_y_displacements = filter(x -> x > 0, map(o -> (o.origin.y - object.origin.y), objects_of_min_distance))

    # @show min_distance 
    # @show objects_of_min_distance 
    # @show positive_y_displacements 

    if length(positive_y_displacements) > 0
      Position(0, unit_size)
    else
      Position(0, 0)
    end
  end
end

function mapPositions(constructor, GRID_SIZE, filterFunction, args, state::Union{State, Nothing}=nothing)::AbstractArray
  map(pos -> constructor(args..., pos), filter(filterFunction, allPositions(GRID_SIZE)))
end

function allPositions(GRID_SIZE, state::Union{State, Nothing}=nothing)
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

function updateOrigin(object::Object, new_origin::Position, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :origin, new_origin)
  new_object
end

function updateAlive(object::Object, new_alive::Bool, state::Union{State, Nothing}=nothing)::Object
  new_object = deepcopy(object)
  new_object = update_nt(new_object, :alive, new_alive)
  new_object
end

function nextLiquid(object::Object, @nospecialize(state::State))::Object
  # # println("nextLiquid")
  GRID_SIZE = state.histories[:GRID_SIZE][0]
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

function nextSolid(object::Object, @nospecialize(state::State))::Object 
  # # println("nextSolid")
  new_object = deepcopy(object)
  if (isWithinBounds(move(object, Position(0, 1)), state) && reduce(&, map(x -> isFree(x, object, state), map(cell -> move(cell.position, Position(0, 1)), render(object, state)))))
    new_object = update_nt(new_object, :origin, move(object.origin, Position(0, 1)))
  end
  new_object
end

function closest(object::Object, positions::Array{Position}, state::Union{State, Nothing}=nothing)::Position
  closestDistance = sort(map(pos -> distance(pos, object.origin), positions))[1]
  closest = filter(pos -> distance(pos, object.origin) == closestDistance, positions)[1]
  closest
end

function isFree(start::Position, stop::Position, @nospecialize(state::State))::Bool 
  GRID_SIZE = state.histories[:GRID_SIZE][0]
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

function isFree(start::Position, stop::Position, object::Object, @nospecialize(state::State))::Bool 
  GRID_SIZE = state.histories[:GRID_SIZE][0]
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

function isFree(position::Position, object::Object, @nospecialize(state::State))
  length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, 
  renderScene(Scene(filter(obj -> obj.id != object.id , state.scene.objects), state.scene.background), state))) == 0
end

function isFree(object::Object, @nospecialize(orig_object::Object), @nospecialize(state::State))::Bool
  reduce(&, map(x -> isFree(x, orig_object, state), map(cell -> cell.position, render(object, state))))
end

function allPositions(@nospecialize(state::State))
  GRID_SIZE = state.histories[:GRID_SIZE][0]
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

function unfold(A, state::Union{State, Nothing}=nothing)
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
# update(@nospecialize(Γ::Env), x::Symbol, v) = merge(Γ, NamedTuple{(x,)}((v,)))

# function update(@nospecialize(Γ::Env), x::Symbol, @nospecialize(v)) 
#   merge(Γ, NamedTuple{(x,)}((v,)))
# end

function update(Γ::Env, x::Symbol, v)::Env 
  setfield!(Γ, x, v)
  Γ
end

function update(Γ::State, x::Symbol, v)::State 
  setfield!(Γ, x, v)
  Γ
end

function update(Γ::Scene, x::Symbol, v)::Scene 
  setfield!(Γ, x, v)
  Γ
end

function update(Γ::Object, x::Symbol, v)::Object 
  if x == :id 
    Γ = @set Γ.id = v
  elseif x == :type 
    Γ = @set Γ.type = v
  elseif x == :alive 
    Γ = @set Γ.alive = v
  elseif x == :changed 
    Γ = @set Γ.changed = v
  elseif x == :custom_fields 
    Γ = @set Γ.custom_fields = v
  elseif x == :render
    Γ = @set Γ.render = v
  elseif x == :origin 
    Γ = @set Γ.origin = v
  else
    # println("yeet")
    Γ = deepcopy(Γ)
    Γ.custom_fields[x] = v
  end
  Γ
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

function primapl(f, x, @nospecialize(Γ::Env)) 
  prim_to_func[f](x), Γ
end

function primapl(f, x1, x2, @nospecialize(Γ::Env))
  prim_to_func[f](x1, x2), Γ
end

lib_to_func = Dict(:Position => Position,
                   :Cell => Cell,
                   :Click => Click,
                   :render => render, 
                   :renderScene => renderScene, 
                   :occurred => occurred,
                   :uniformChoice => uniformChoice, 
                   :min => min,
                   :isWithinBounds => isWithinBounds, 
                   :isOutsideBounds => isOutsideBounds,
                   :clicked => clicked, 
                   :objClicked => objClicked, 
                   :intersects => intersects, 
                   :moveIntersects => moveIntersects,
                   :pushConfiguration => pushConfiguration,
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
                   :adj => adj,
                  #  :rotate => rotate, 
                  #  :rotateNoCollision => rotateNoCollision, 
                   :move => move, 
                   :moveLeft => moveLeft, 
                   :moveRight => moveRight, 
                   :moveUp => moveUp, 
                   :moveDown => moveDown, 
                   :moveNoCollision => moveNoCollision, 
                   :moveNoCollisionColor => moveNoCollisionColor, 
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
                   :closestRandom => closestRandom,
                   :closestLeft => closestLeft,
                   :closestRight => closestRight,
                   :closestUp => closestUp,
                   :closestDown => closestDown, 
                   :farthestRandom => farthestRandom,
                   :farthestLeft => farthestLeft,
                   :farthestRight => farthestRight,
                   :farthestUp => farthestUp,
                   :farthestDown => farthestDown, 
                   :mapPositions => mapPositions, 
                   :allPositions => allPositions, 
                   :updateOrigin => updateOrigin, 
                   :updateAlive => updateAlive, 
                   :nextLiquid => nextLiquid, 
                   :nextSolid => nextSolid,
                   :unfold => unfold,
                   :prev => prev,
                   :firstWithDefault => firstWithDefault,
                  )
islib(f) = f in keys(lib_to_func)

# library function handling 
function libapl(f, args, @nospecialize(Γ::Env))
  # println("libapl")
  # @show f 
  # @show args 

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
      elseif f == :removeObj 
        interpret_removeObj(args, Γ)
      else 
        lib_to_func[f](map(x -> interpret(x, Γ)[1], args)..., Γ.state), Γ
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

function julialibapl(f, args, @nospecialize(Γ::Env))
  if !(f in [:map, :filter])
    julia_lib_to_func[f](args...), Γ
  elseif f == :map 
    interpret_julia_map(args, Γ)
  elseif f == :filter 
    interpret_julia_filter(args, Γ)
  end
end

function interpret(aex::AExpr, @nospecialize(Γ::Env))
  arr = [aex.head, aex.args...]
  # # # # # # println()
  # # # # # # println("Env:")
  # display(Γ)
  # # # # # # @showarr 
  # next(x) = interpret(x, Γ)
  isaexpr(x) = x isa AExpr
  t = MLStyle.@match arr begin
    [:if, c, t, e]                                                    => let (v, Γ2) = interpret(c, Γ) 
                                                                            if v == true
                                                                              interpret(t, Γ2)
                                                                            else
                                                                              interpret(e, Γ2)
                                                                            end
                                                                        end
    [:assign, x, v::AExpr] && if v.head == :initnext end              => interpret_init_next(x, v, Γ)
    [:assign, x, v::Union{AExpr, Symbol}]                             => let (v2, Γ_) = interpret(v, Γ)
                                                                          # # # @showv 
                                                                          # # # @showx
                                                                          interpret(AExpr(:assign, x, v2), Γ_)
                                                                         end
    [:assign, x, v]                                                   => let
                                                                          # # @showx 
                                                                          # # @showv 
                                                                          if x in fieldnames(typeof(Γ))
                                                                            # # println("here") 
                                                                            # # @showΓ[x]
                                                                          end
                                                                          # # @showupdate(Γ, x, v)[x]
                                                                          # # println("returning")
                                                                          # @show v isa AbstractArray
                                                                          Γ.current_var_values[x] = v isa BigInt ? convert(Int, v) : v
                                                                          (aex, Γ) 
                                                                         end
    [:list, args...]                                                  => interpret_list(args, Γ)
    [:typedecl, args...]                                              => (aex, Γ)
    [:let, args...]                                                   => interpret_let(args, Γ) 
    [:lambda, args...]                                                => (args, Γ)
    [:fn, args...]                                                    => (args, Γ)
    [:call, f, arg1] && if isprim(f) end                              => let (new_arg, Γ2) = interpret(arg1, Γ)
                                                                             primapl(f, new_arg, Γ2)
                                                                         end
                                                                    
    [:call, f, arg1, arg2] && if isprim(f) end                        => let (new_arg1, Γ2) = interpret(arg1, Γ)
                                                                             (new_arg2, Γ2) = interpret(arg2, Γ2)
                                                                             primapl(f, new_arg1, new_arg2, Γ2)
                                                                         end
    [:call, f, args...] && if f == :prev && args != [:obj] end        => interpret(AExpr(:call, Symbol(string(f, uppercasefirst(string(args[1])))), :state), Γ)
    [:call, f, args...] && if islib(f) end                            => interpret_lib(f, args, Γ)
    [:call, f, args...] && if isjulialib(f) end                       => interpret_julia_lib(f, args, Γ)
    [:call, f, args...] && if f in keys(Γ.state.object_types) end     => interpret_object_call(f, args, Γ)
    [:call, f, args...]                                               => interpret_call(f, args, Γ)
     
    [:field, x, fieldname]                                            => interpret_field(x, fieldname, Γ)
    [:object, args...]                                                => interpret_object(args, Γ)
    [:on, args...]                                                    => interpret_on(args, Γ)
    [args...]                                                         => error(string("Invalid AExpr Head: ", aex.head))
    _                                                                 => error("Could not interpret $arr")
  end
  # # # # # # println("FINSIH", arr)
  # # # # # @show(t)
  # # println("T[2]")
  # # @showt[2]
  t
end

function interpret(x::Symbol, @nospecialize(Γ::Env))
  if x == Symbol("false")
    false, Γ
  elseif x == Symbol("true")
    true, Γ
  elseif x == :left 
    Γ.left, Γ
  elseif x == :right 
    Γ.right, Γ
  elseif x == :up
    Γ.up, Γ
  elseif x == :down 
    Γ.down, Γ
  elseif x == :click 
    Γ.click, Γ
  elseif x == :clicked 
    interpret(AExpr(:call, :occurred, :click), Γ)
  elseif x in keys(Γ.state.object_types)
    x, Γ
  elseif x == :state 
    Γ.state, Γ
  elseif x in keys(Γ.current_var_values)
    # # # @showeval(:($(Γ).$(x)))
    Γ.current_var_values[x], Γ
  else
    error("Could not interpret $x")
  end
end

# if x is not an AExpr or a Symbol, it is just a value (return it)
function interpret(x, @nospecialize(Γ::Env))
  if x isa BigInt 
    (convert(Int, x), Γ)
  else
    (x, Γ)
  end
end 

function interpret_list(args, @nospecialize(Γ::Env))
  new_list = []
  for arg in args
    new_arg, Γ = interpret(arg, Γ)
    push!(new_list, new_arg)
  end
  new_list, Γ
end

function interpret_lib(f, args, @nospecialize(Γ::Env))
  # println("INTERPRET_LIB")
  # @show f 
  # @show args 
  new_args = []
  for arg in args 
    new_arg, Γ = interpret(arg, Γ)
    push!(new_args, new_arg)
  end
  # # # @shownew_args
  libapl(f, new_args, Γ)
end

function interpret_julia_lib(f, args, @nospecialize(Γ::Env))
  # println("INTERPRET_JULIA_LIB")
  # @show f 
  # @show args
  new_args = []
  for i in 1:length(args)
    arg = args[i] 
    # # # @showarg
    if f == :get && i == 2 && args[i] isa Symbol
      new_arg = arg
    else
      new_arg, Γ = interpret(arg, Γ)
    end
    # # # @shownew_arg 
    # # # @showΓ
    push!(new_args, new_arg)
  end
  # @show new_args 
  julialibapl(f, new_args, Γ)
end

function interpret_field(x, f, @nospecialize(Γ::Env))
  # # # # println("INTERPRET_FIELD")
  # # # # @showkeys(Γ)
  # # # # @showx 
  # # # # @showf 
  val, Γ2 = interpret(x, Γ)
  if val isa Object
    if f in [:id, :origin, :type, :alive, :changed, :render]
      (getfield(val, f), Γ2)
    else
      (val.custom_fields[f], Γ2)
    end
  else
    (getfield(val, f), Γ2)
  end
end

function interpret_let(args::AbstractArray, @nospecialize(Γ::Env))
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

# used for lambda function calls!
function interpret_call(f, params, @nospecialize(Γ::Env))
  func, Γ = interpret(f, Γ)
  func_args = func[1]
  func_body = func[2]

  # construct environment
  old_current_var_values = copy(Γ.current_var_values) 
  Γ2 = Γ
  if func_args isa AExpr 
    for i in 1:length(func_args.args)
      param_name = func_args.args[i]
      param_val, Γ2 = interpret(params[i], Γ2)
      Γ2.current_var_values[param_name] = param_val
    end
  elseif func_args isa Symbol
    param_name = func_args
    param_val, Γ2 = interpret(params[1], Γ2)
    Γ2.current_var_values[param_name] = param_val
  else
    error("Could not interpret $(func_args)")
  end
  # # # # # @showtypeof(Γ2)
  # evaluate func_body in environment 
  v, Γ2 = interpret(func_body, Γ2)
  
  # return value and original environment, except with state updated 
  Γ = update(Γ, :state, update(Γ.state, :objectsCreated, Γ2.state.objectsCreated))
  # # # # # println("DONE")
  Γ.current_var_values = old_current_var_values
  (v, Γ)
end

function interpret_object_call(f, args, @nospecialize(Γ::Env))
  # # # # # println("BEFORE")
  # # # # # @showΓ.state.objectsCreated 
  new_state = update(Γ.state, :objectsCreated, Γ.state.objectsCreated + 1)
  Γ = update(Γ, :state, new_state)

  origin, Γ = interpret(args[end], Γ)
  # object_repr = (origin=origin, type=f, alive=true, changed=false, id=Γ.state.objectsCreated)

  old_current_var_values = copy(Γ.current_var_values)
  Γ2 = Γ
  fields = Γ2.state.object_types[f].fields
  field_values = Dict()
  for i in 1:length(fields)
    field_name = fields[i].args[1]
    field_value, Γ2 = interpret(args[i], Γ2)
    field_values[field_name] = field_value
    # object_repr = update(object_repr, field_name, field_value)
    Γ2.current_var_values[field_name] = field_value
  end
  # @show f
  # @show field_values 
  if length(fields) == 0 
    object_repr = Object(Γ.state.objectsCreated, origin, f, true, false, field_values, nothing)  
  else
    render, Γ2 = interpret(Γ.state.object_types[f].render, Γ2)
    render = render isa AbstractArray ? render : [render]
    object_repr = Object(Γ.state.objectsCreated, origin, f, true, false, field_values, render)
  end
  Γ.current_var_values = old_current_var_values
  # # # # # println("AFTER")
  # # # # # @showΓ.state.objectsCreated 
  (object_repr, Γ)  
end

function interpret_init_next(var_name, var_val, @nospecialize(Γ::Env))
  # # # println("INTERPRET INIT NEXT")
  init_func = var_val.args[1]
  next_func = var_val.args[2]

  Γ2 = Γ
  if !(var_name in keys(Γ2.current_var_values)) # variable not initialized; use init clause
    # # # println("HELLO")
    # initialize var_name
    var_val, Γ2 = interpret(init_func, Γ2)
    Γ2.current_var_values[var_name] = var_val

    # construct history variable in state 
    Γ2.state.histories[Symbol(string(var_name))] = Dict()
    # Γ2 = update(Γ2, :state, new_state)

    # construct prev function 
    _, Γ2 = interpret(AExpr(:assign, Symbol(string(:prev, uppercasefirst(string(var_name)))), parseautumn("""(fn (state) (get (get (.. state histories) $(string(var_name)) -1) (- (.. state time) 1) $(var_name)))""")), Γ2) 

  elseif Γ.state.time > 0 # variable initialized; use next clause if simulation time > 0  
    # update var_val 
    var_val = Γ.current_var_values[var_name]
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
      if var_name in keys(Γ.on_clauses)
        events = Γ.on_clauses[var_name]
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
    Γ2.current_var_values[var_name] = final_val
  end
  (AExpr(:assign, var_name, var_val), Γ2)
end

function interpret_object(args, @nospecialize(Γ::Env))
  object_name = args[1]
  object_fields = args[2:end-1]
  object_render = args[end]

  # construct object creation function
  if length(object_fields) == 0
    render, _ = interpret(object_render, Γ)
    if !(render isa AbstractArray) 
      render = [render]
    end
    object_tuple = ObjectType(render, object_fields)
  else
    object_tuple = ObjectType(object_render, object_fields)
  end
  Γ.state.object_types[object_name] = object_tuple
  (AExpr(:object, args...), Γ)
end

function interpret_on(args, @nospecialize(Γ::Env))
  # println("INTERPRET ON")
  event = args[1]
  update_ = args[2]
  # @show event 
  # @show update_
  Γ2 = Γ
  if Γ2.state.time == 0 
    if update_.head == :assign
      var_name = update_.args[1]
      if !(var_name in keys(Γ2.on_clauses))
        Γ2.on_clauses[var_name] = [event]
      else
        Γ2.on_clauses[var_name] = vcat(event, Γ2.on_clauses[var_name])
      end
    elseif update_.head == :let 
      assignments = update_.args
      if length(assignments) > 0 
        if (assignments[end] isa AExpr) && (assignments[end].head == :assign)
          for a in assignments 
            var_name = a.args[1]
            if !(var_name in keys(Γ2.on_clauses))
              Γ2.on_clauses[var_name] = [event]
            else
              Γ2.on_clauses[var_name] = vcat(event, Γ2.on_clauses[var_name])
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
function interpret_updateObj(args, @nospecialize(Γ::Env))
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
    object_type = Γ.state.object_types[obj.type]
    
    old_current_var_values = copy(Γ.current_var_values)
    Γ3 = Γ2
    fields = object_type.fields
    for i in 1:length(fields)
      field_name = fields[i].args[1]
      field_value = new_obj.custom_fields[field_name]
      Γ3.current_var_values[field_name] = field_value
    end

    if length(fields) != 0 
      render, Γ3 = interpret(Γ.state.object_types[obj.type].render, Γ3)
      render = render isa AbstractArray ? render : [render]
      new_obj = update(new_obj, :render, render)
    end  
    Γ2.current_var_values = old_current_var_values
    new_obj, Γ2
    # Γ2 = update(Γ2, :state, update(Γ2.state, :objectsCreated, Γ2.state.objectsCreated + 1))
  else
    error("Could not interpret updateObj")
  end
end

function interpret_removeObj(args, @nospecialize(Γ::Env))
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

function interpret_julia_map(args, @nospecialize(Γ::Env))
  new_list = []
  map_func = args[1]
  list, Γ = interpret(args[2], Γ)
  for arg in list  
    new_arg, Γ = interpret(AExpr(:call, map_func, arg), Γ)
    push!(new_list, new_arg)
  end
  new_list, Γ
end

function interpret_julia_filter(args, @nospecialize(Γ::Env))
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

function interpret_program(aex, Γ::Env)
  aex.head == :program || error("Must be a program aex")
  for line in aex.args
    v, Γ = interpret(line, Γ)
  end
  return aex, Γ
end

function start(aex::AExpr, rng=Random.GLOBAL_RNG)
  aex.head == :program || error("Must be a program aex")
  # env = (on_clauses=empty_env(),
  #        left=false, 
  #        right=false,
  #        up=false,
  #        down=false,
  #        click=nothing, 
  #        state=(time=0, objectsCreated=0, rng=rng, scene=empty_env(), object_types=empty_env()))

  env = Env(false, false, false, false, nothing, Dict(), Dict(), Dict(), State(0, 0, rng, Scene([], "white"), Dict(), Dict()))

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
    # new_state = update(env.state, Symbol(string(var_name, "History")), Dict())
    env.state.histories[var_name] = Dict()
    # env = update(env, :state, new_state)

    # construct prev function 
    _, env = interpret(AExpr(:assign, Symbol(string(:prev, uppercasefirst(string(var_name)))), parseautumn("""(fn () (get (.. (.. state histories) $(string(var_name))) (- (.. state time) 1) $(var_name)))""")), env) 
  end

  # add background to scene 
  background_assignments = filter(l -> l.args[1] == :background, lifted_lines)
  background = background_assignments != [] ? background_assignments[end].args[2] : "#ffffff00"
  env.state.scene.background = background


  # initialize scene.objects 
  # env = update(env, :state, update(env.state, :scene, update(env.state.scene, :objects, [])))

  # initialize lifted variables
  # env = update(env, :lifted, empty_env()) 
  for line in lifted_lines
    var_name = line.args[1]
    env.lifted[var_name] = line.args[2] 
    if var_name in [:GRID_SIZE, :background]
      env.current_var_values[var_name] = interpret(line.args[2], env)[1]
    end
  end 

  new_aex = AExpr(:program, reordered_lines_temp...) # try interpreting the init_next's before on for the first time step (init)
  # # @show new_aex
  aex_, env_ = interpret_program(new_aex, env)

  # update state (time, histories, scene)
  env_ = update_state(env_)

  AExpr(:program, reordered_lines...), env_
end

function step(aex::AExpr, env::Env, user_events=(click=nothing, left=false, right=false, down=false, up=false))::Env
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
function update_state(env_::Env)
  # reset user events 
  for user_event in [:left, :right, :up, :down]
    env_ = update(env_, user_event, false)
  end
  env_ = update(env_, :click, nothing)

  # add updated variable values to history
  for key in keys(env_.state.histories)    
    env_.state.histories[key][env_.state.time] = env_.current_var_values[key]
  
    # delete earlier times stored in history, since we only use prev up to 1 level back
    if env_.state.time > 0
      delete!(env_.state.histories, env_.state.time - 1)
    end

  end

  # update lifted variables 
  for var_name in keys(env_.lifted)
    env_.current_var_values[var_name] = interpret(env_.lifted[var_name], env_)[1]
  end

  # update scene.objects 
  new_scene_objects = []
  for key in keys(env_.current_var_values)
    if ((env_.current_var_values[key] isa Object) || (env_.current_var_values[key] isa AbstractArray && (length(env_.current_var_values[key]) > 0) && (env_.current_var_values[key][1] isa Object)))
      object_val = env_.current_var_values[key]
      if object_val isa AbstractArray 
        push!(new_scene_objects, object_val...)
      else
        push!(new_scene_objects, object_val)
      end
    end
  end
  env_.state.scene.objects = new_scene_objects

  # update time 
  new_state = update(env_.state, :time, env_.state.time + 1)
  env_ = update(env_, :state, new_state)
end

function interpret_over_time(aex::AExpr, iters, user_events=[])::Env
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

function interpret_over_time_variable(aex::AExpr, var_name, iters, user_events=[])
  variable_values = []
  new_aex, env_ = start(aex)
  push!(variable_values, env_.state.histories[var_name][env_.state.time])
  if user_events == []
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_)
      push!(variable_values, env_.state.histories[var_name][env_.state.time])
    end
  else
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_, user_events[i])
      push!(variable_values, env_.state.histories[var_name][env_.state.time])
    end
  end
  variable_values
end

function interpret_over_time_observations(aex::AExpr, iters, user_events=[])
  scenes = []
  new_aex, env_ = start(aex)
  push!(scenes, AutumnStandardLibrary.renderScene(env_.state.scene, env_.state))
  if user_events == []
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_)
      push!(scenes, AutumnStandardLibrary.renderScene(env_.state.scene, env_.state))
    end
  else
    for i in 1:iters
      # # @show i
      env_ = step(new_aex, env_, user_events[i])
      push!(scenes, AutumnStandardLibrary.renderScene(env_.state.scene, env_.state))
    end
  end
  scenes
end

end
