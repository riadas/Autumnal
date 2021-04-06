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
import Base.min
using SExpressions

MODS = Dict{Int, Any}(); 
HISTORY = Dict{Int, Dict{Int, Any}}()

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
      # println("PARSED SUCCESSFULLY")
      compiledAutumn = compiletojulia(parsedAutumn)
      # println("COMPILED SUCCESSFULLY")
      mod = eval(compiledAutumn)
      # println("EVALUATED SUCCESSFULLY")
      content = showstring(parsedAutumn)
      MODS[clientid] = mod
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
  # println("step")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  # println(state.time)
  next_state = mod.next(state, nothing, nothing, nothing, nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
end

function startautumn()
  # println("startautumn")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.init(nothing, nothing, nothing, nothing, nothing)
  # println(state.time)
  grid_size = state.GRID_SIZEHistory[state.time]
  background = :backgroundHistory in fieldnames(typeof(state)) ? state.backgroundHistory[state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(state.scene))
  HISTORY[clientid] = Dict{Int, Any}()
  HISTORY[clientid][state.time] = mod.render(state.scene)
  json(vcat(grid_size, background, cells))
  # json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].init(nothing)) : []))
end

function clicked()
  # println("click")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  # println(state.time)
  next_state = mod.next(state, mod.Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))), nothing, nothing, nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
  #json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].next(MODS[clientid].Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))))) : []))
end

function up()
  # println("up")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  next_state = mod.next(state, nothing, nothing, nothing, mod.Up(), nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
end

function down()
  # println("down")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  next_state = mod.next(state, nothing, nothing, nothing, nothing, mod.Down())
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
end

function right()
  # println("right")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  next_state = mod.next(state, nothing, nothing, mod.Right(), nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
end

function left()
  # println("left")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.state
  next_state = mod.next(state, nothing, mod.Left(), nothing, nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  HISTORY[clientid][next_state.time] = mod.render(next_state.scene)
  json(vcat(background, cells))
end

function replay()
  # println("replay")
  clientid = parse(Int64, @params(:clientid))
  json(HISTORY[clientid])
  # json(Dict(key => map(particle -> [particle.position.x, particle.position.y, particle.color], filter(particle -> particle.render, particles)) for (key, particles) in history))
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

"Autumn Compile Error"
struct AutumnCompileError <: Exception
  msg
end
AutumnCompileError() = AutumnCompileError("")
abstract type Object end
# ----- Compile Helper Functions ----- #

function compile(expr::AExpr, data::Dict{String, Any}, parent::Union{AExpr, Nothing}=nothing)
  arr = [expr.head, expr.args...]
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
    [args...] => throw(AutumnCompileError(string("Invalid AExpr Head: ", expr.head))) # if expr head is not one of the above, throw error
  end
end

function compile(expr::AbstractArray, data::Dict{String, Any}, parent::Union{AExpr, Nothing}=nothing)
  if length(expr) == 0 || (length(expr) > 1 && expr[1] != :List)
    throw(AutumnCompileError("Invalid List Syntax"))
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
  if (typeof(expr.args[2]) == AExpr && expr.args[2].head == :fn)
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
    # println(expr.args[1])
    # println(expr.args[2])
    data["types"][expr.args[1]] = expr.args[2]
    :()
  else
    :(local $(compile(expr.args[1], data))::$(compile(expr.args[2], data)))
  end
end

function compileexternal(expr::AExpr, data::Dict{String, Any})
  # println("here: ")
  # println(expr.args[1])
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
  elseif !(fnName in binaryOperators) && fnName != :prev
    :($(fnName)($(map(x -> compile(x, data), expr.args[2:end])...)))
  elseif fnName == :prev
    :($(Symbol(string(expr.args[2]) * "Prev"))($(map(compile, expr.args[3:end])...)))
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
      hidden::Bool
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
  # println("here")
  # println(typeof(expr.args[1]) == AExpr ? expr.args[1].args[1] : expr.args[1])
  # event = compile(expr.args[1], data)
  # response = compile(expr.args[2], data)
  event = expr.args[1]
  response = expr.args[2]

  push!(data["on"], (event, response))
  :()
end

function compileinitnext(data::Dict{String, Any})
  init = quote
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), data["lifted"])...)
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2].args[1], data))), data["initnext"])...)
  end

  onClauses = map(x -> quote 
    if $(compile(x[1], data))
      $(compile(x[2], data))
    end
  end, data["on"])
  notOnClauses = map(x -> quote 
                            if !(foldl(|, [$(map(y -> compile(y[1], data), filter(z -> ((z[2].head == :assign) && (z[2].args[1] == x.args[1])) || ((z[2].head == :let) && (x.args[1] in map(w -> w.args[1], z[2].args))), data["on"]))...)]; init=false))
                              $(compile(x.args[1], data)) = $(compile(x.args[2].args[2], data));                                 
                            end
                          end, data["initnext"])

  next = quote
    $(map(x -> :($(compile(x.args[1], data)) = state.$(Symbol(string(x.args[1])*"History"))[state.time - 1]), 
      vcat(data["initnext"], data["lifted"]))...)
    $(onClauses...)
    $(notOnClauses...)
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))...)
  end

  initFunction = quote
    function init($(map(x -> :($(x.args[1])::Union{$(compile(data["types"][x.args[1]], data)), Nothing}), data["external"])...))::STATE
      $(compileinitstate(data))
      $(init)
      $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))...)
      $(map(x -> :(state.$(Symbol(string(x.args[1])*"History"))[state.time] = $(x.args[1])), 
            vcat(data["external"], data["initnext"], data["lifted"]))...)
            state.scene = Scene(vcat([$(filter(x -> get(data["types"], x, :Any) in vcat(data["objects"], map(x -> [:List, x], data["objects"])), 
        map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))...)]...), :backgroundHistory in fieldnames(STATE) ? state.backgroundHistory[state.time] : "#ffffff00")
      
      global state = deepcopy(state)
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
      global state = deepcopy(state)
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
                          click !== nothing
                        end
                      end,
"uniformChoice"   =>  quote
                        function uniformChoice(freePositions)
                          freePositions[rand(Categorical(ones(length(freePositions))/length(freePositions)))]
                        end
                      end,
"uniformChoice2"   =>  quote
                        function uniformChoice(freePositions, n)
                          map(idx -> freePositions[idx], rand(Categorical(ones(length(freePositions))/length(freePositions)), n))
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

                    function render(scene::Scene)::Array{Cell}
                      vcat(map(obj -> render(obj), filter(obj -> obj.alive && !obj.hidden, scene.objects))...)
                    end

                    function render(obj::Object)::Array{Cell}
                      map(cell -> Cell(move(cell.position, obj.origin), cell.color), obj.render)
                    end

                    function isWithinBounds(obj::Object)::Bool
                      # println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
                      length(filter(cell -> !isWithinBounds(cell.position), render(obj))) == 0
                    end

                    function clicked(click::Union{Click, Nothing}, object::Object)::Bool
                      if click == nothing
                        false
                      else
                        GRID_SIZE = state.GRID_SIZEHistory[0]
                        nums = map(cell -> GRID_SIZE*cell.position.y + cell.position.x, render(object))
                        (GRID_SIZE * click.y + click.x) in nums
                      end
                    end

                    function clicked(click::Union{Click, Nothing}, objects::AbstractArray)
                      # println("LOOK AT ME")
                      # println(reduce(&, map(obj -> clicked(click, obj), objects)))
                      reduce(|, map(obj -> clicked(click, obj), objects))
                    end

                    function objClicked(click::Union{Click, Nothing}, objects::AbstractArray)::Object
                      # println(click)
                      filter(obj -> clicked(click, obj), objects)[1]
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
                      nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
                      nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj2))
                      length(intersect(nums1, nums2)) != 0
                    end

                    function intersects(obj1::Object, obj2::Array{<:Object})::Bool
                      nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, render(obj1))
                      nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                      length(intersect(nums1, nums2)) != 0
                    end

                    function intersects(obj1::Array{<:Object}, obj2::Array{<:Object})::Bool
                      nums1 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj1)...))
                      nums2 = map(cell -> state.GRID_SIZEHistory[0]*cell.position.y + cell.position.x, vcat(map(render, obj2)...))
                      length(intersect(nums1, nums2)) != 0
                    end

                    function intersects(list1, list2)::Bool
                      length(intersect(list1, list2)) != 0 
                    end

                    function intersects(object::Object)::Bool
                      objects = state.scene.objects
                      intersects(object, objects)
                    end

                    function addObj(list::Array{<:Object}, obj::Object)
                      new_list = vcat(list, obj)
                      new_list
                    end

                    function addObj(list::Array{<:Object}, objs::Array{<:Object})
                      new_list = vcat(list, objs)
                      new_list
                    end

                    function removeObj(list::Array{<:Object}, obj::Object)
                      filter(x -> x.id != obj.id, list)
                    end

                    function removeObj(list::Array{<:Object}, fn)
                      orig_list = filter(obj -> !fn(obj), list)
                    end

                    function removeObj(obj::Object)
                      obj.alive = false
                      deepcopy(obj)
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
                      setproperty!(new_obj, :hidden, obj.hidden)

                      setproperty!(new_obj, Symbol(field), value)  
                      new_obj
                    end

                    function filter_fallback(obj::Object)
                      true
                    end

                    function updateObj(list::Array{<:Object}, map_fn, filter_fn=filter_fallback)
                      orig_list = filter(obj -> !filter_fn(obj), list)
                      filtered_list = filter(filter_fn, list)
                      new_filtered_list = map(map_fn, filtered_list)
                      vcat(orig_list, new_filtered_list)                     
                    end

                    function adjPositions(position::Position)::Array{Position}
                      filter(isWithinBounds, [Position(position.x, position.y + 1), Position(position.x, position.y - 1), Position(position.x + 1, position.y), Position(position.x - 1, position.y)])
                    end

                    function adjPositions(obj::Object)
                      neighborPositions = vcat(map(cell -> adjPositions(cell.position), render(obj))...)
                      unique(filter(pos -> !(pos in render(obj)), neighborPositions))
                    end

                    function isWithinBounds(position::Position)::Bool
                      (position.x >= 0) && (position.x < state.GRID_SIZEHistory[0]) && (position.y >= 0) && (position.y < state.GRID_SIZEHistory[0])                          
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

                    function unitVector(position1::Position, position2::Position)::Position
                      deltaX = position2.x - position1.x
                      deltaY = position2.y - position1.y
                      if (floor(Int, abs(sign(deltaX))) == 1 && floor(Int, abs(sign(deltaY))) == 1)
                        uniformChoice([Position(sign(deltaX), 0), Position(0, sign(deltaY))])
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
                      # println("hello")
                      # println(Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE))
                      Position((position.x + x + GRID_SIZE) % GRID_SIZE, (position.y + y + GRID_SIZE) % GRID_SIZE)
                    end

                    # ----- begin left/right moveWrap ----- #

                    function moveLeftWrap(object::Object)::Object
                      new_object = deepcopy(object)
                      new_object.position = moveWrap(object.origin, -1, 0)
                      new_object
                    end
                      
                    function moveRightWrap(object::Object)::Object
                      new_object = deepcopy(object)
                      new_object.position = moveWrap(object.origin, 1, 0)
                      new_object
                    end

                    function moveUpWrap(object::Object)::Object
                      new_object = deepcopy(object)
                      new_object.position = moveWrap(object.origin, 0, -1)
                      new_object
                    end

                    function moveDownWrap(object::Object)::Object
                      new_object = deepcopy(object)
                      new_object.position = moveWrap(object.origin, 0, 1)
                      new_object
                    end

                    # ----- end left/right moveWrap ----- #

                    function randomPositions(GRID_SIZE::Int, n::Int)::Array{Position}
                      nums = uniformChoice([0:(GRID_SIZE * GRID_SIZE - 1);], n)
                      # println(nums)
                      # println(map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums))
                      map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
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

                    function mapPositions(constructor, GRID_SIZE::Int, filterFunction, args...)::Union{Object, Array{<:Object}}
                      map(pos -> constructor(args..., pos), filter(filterFunction, allPositions(GRID_SIZE)))
                    end

                    function allPositions(GRID_SIZE::Int)
                      nums = [0:(GRID_SIZE * GRID_SIZE - 1);]
                      map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
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
                      # println("nextLiquid")
                      GRID_SIZE = state.GRID_SIZEHistory[0]
                      new_object = deepcopy(object)
                      if object.origin.y != GRID_SIZE - 1 && isFree(move(object.origin, Position(0, 1)))
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
                      # println("nextSolid")
                      GRID_SIZE = state.GRID_SIZEHistory[0] 
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
                      nums = [(GRID_SIZE * start.y + start.x):(GRID_SIZE * stop.y + stop.x);]
                      reduce(&, map(num -> isFree(Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE))), nums))
                    end

                    function isFree(start::Position, stop::Position, object::Object)::Bool 
                      GRID_SIZE = state.GRID_SIZEHistory[0]
                      nums = [(GRID_SIZE * start.y + start.x):(GRID_SIZE * stop.y + stop.x);]
                      reduce(&, map(num -> isFree(Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), object), nums))
                    end

                    function isFree(position::Position, object::Object)
                      length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, 
                      filter(x -> !(x in render(object)), render(state.scene)))) == 0
                    end

                    function isFree(object::Object, orig_object::Object)::Bool
                      reduce(&, map(x -> isFree(x, orig_object), map(cell -> cell.position, render(object))))
                    end

                    function allPositions()
                      GRID_SIZE = state.GRID_SIZEHistory[0]
                      nums = [1:GRID_SIZE*GRID_SIZE - 1;]
                      map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums)
                    end
                  
                end
])

# binary operators
const binaryOperators = [:+, :-, :/, :*, :&, :|, :>=, :<=, :>, :<, :(==), :!=, :%, :&&]


# compile.jl
 
"compile `aexpr` into Expr"
function compiletojulia(aexpr::AExpr)::Expr

  # dictionary containing types/definitions of global variables, for use in constructing init func.,
  # next func., etcetera; the three categories of global variable are external, initnext, and lifted  
  historydata = Dict([("external" => [au"""(external (: click Click))""".args[1], au"""(external (: left KeyPress))""".args[1], au"""(external (: right KeyPress))""".args[1], au"""(external (: up KeyPress))""".args[1], au"""(external (: down KeyPress))""".args[1]]), # :typedecl aexprs for all external variables
               ("initnext" => []), # :assign aexprs for all initnext variables
               ("lifted" => []), # :assign aexprs for all lifted variables
               ("types" => Dict{Symbol, Any}([:click => :Click, :left => :KeyPress, :right => :KeyPress, :up => :KeyPress, :down => :KeyPress, :GRID_SIZE => :Int, :background => :String])), # map of global variable names (symbols) to types
               ("on" => []),
               ("objects" => [])]) 
               
  if (aexpr.head == :program)
    # handle AExpression lines
    lines = filter(x -> x !== :(), map(arg -> compile(arg, historydata, aexpr), aexpr.args))
    
    # construct STATE struct and initialize state::STATE
    stateStruct = compilestatestruct(historydata)
    initStateStruct = compileinitstate(historydata)
    
    # handle init, next, prev, and built-in functions
    initnextFunctions = compileinitnext(historydata)
    prevFunctions = compileprevfuncs(historydata)
    builtinFunctions = compilebuiltin()

    # remove empty lines
    lines = filter(x -> x != :(), 
            vcat(builtinFunctions, lines, stateStruct, initStateStruct, prevFunctions, initnextFunctions))

    # construct module
    expr = quote
      module CompiledProgram
        export init, next
        import Base.min
        using Distributions
        using MLStyle: @match 
        $(lines...)
      end
    end  
    expr.head = :toplevel
    expr
  else
    throw(AutumnCompileError("AExpr Head != :program"))
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

# scene.jl

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
    $(join(map(t -> "(object ObjType$(t.id) (list $(join(map(cell -> """(Cell $(cell[1]) $(cell[2]) "$(t.color)")""", t.shape), " "))))", types), "\n  "))

    $((join(map(obj -> """(: obj$(obj.id) ObjType$(obj.type.id))""", objects), "\n  "))...)

    $((join(map(obj -> """(= obj$(obj.id) (initnext (ObjType$(obj.type.id) (Position $(obj.position[1]) $(obj.position[2]))) (prev obj$(obj.id))))""", objects), "\n  ")))
  )
  """
end

colors = ["red", "yellow", "green", "blue"]
backgroundcolors = ["white", "black"]
function generatescene_program(rng=Random.GLOBAL_RNG; gridsize::Int=16)
  types, objects, background, _ = generatescene_objects(rng, gridsize=gridsize)
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

function neighbors(shape::AbstractArray)
  neighborPositions = vcat(map(pos -> neighbors(pos), shape)...)
  unique(filter(pos -> !(pos in shape), neighborPositions))
end

function neighbors(position)
  x = position[1]
  y = position[2]
  [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
end

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

function color_contiguity_autumn(position_to_color, pos1, pos2)
  length(intersect(position_to_color[pos1], position_to_color[pos2])) > 0
end

function parsescene_autumn(render_output::AbstractArray, dim::Int=16, background::String="white"; color=true)
  position_to_color = Dict()
  for cell in render_output
    if (cell.position.x, cell.position.y) in keys(position_to_color)
      push!(position_to_color[(cell.position.x, cell.position.y)], cell.color)
    else
      position_to_color[(cell.position.x, cell.position.y)] = [cell.color] 
    end
  end

  colored_positions = sort(collect(keys(position_to_color)))
  objectshapes = []
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
          if (n in colored_positions) && !(n in visited) && (color ? color_contiguity_autumn(position_to_color, n, pos) : true) 
            enqueue!(q, n)
          end
        end
      end
      push!(objectshapes, objectshape)
    end
  end  

  types = []
  objects = []
  # @show length(objectshapes)
  for objectshape in objectshapes
    objectcolor = position_to_color[objectshape[1]][1]
    # @show objectcolor 
    # @show objectshape
    translated = map(pos -> dim * pos[2] + pos[1], objectshape)
    translated = length(translated) % 2 == 0 ? translated[1:end-1] : translated # to produce a single median
    centerPos = objectshape[findall(x -> x == median(translated), translated)[1]]
    translatedShape = unique(map(pos -> (pos[1] - centerPos[1], pos[2] - centerPos[2]), objectshape))

    if !((translatedShape, objectcolor) in map(type -> (type.shape, type.color) , types))
      push!(types, ObjType(translatedShape, objectcolor, [], length(types) + 1))
      push!(objects, Obj(types[length(types)], centerPos, [], length(objects) + 1))
    else
      type_id = findall(type -> (type.shape, type.color) == (translatedShape, objectcolor), types)[1]
      push!(objects, Obj(types[type_id], centerPos, [], length(objects) + 1))
    end
  end
  (types, objects, background, dim)
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
      ("moveLeft", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveRight", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveUp", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
      ("moveDown", [:(genObjectUpdateRule($(object), $(environment), p=0.9))]),
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

end
