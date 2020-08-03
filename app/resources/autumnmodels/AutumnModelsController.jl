module AutumnModelsController
using Genie.Renderer
using Genie.Renderer.Html: html
using Genie.Renderer.Json: json
using Genie.Router
using Genie.Requests
using MLStyle
import Base.min
using SExpressions

MODS = Dict{Int, Any}(); 
STATES = Dict{Int, Any}();
HISTORY = Dict{Int, Dict{Int, Any}}()

function autumnmodels()
  redirect(:playground)
end

function playground()
  html(:autumnmodels, :autumnmodelsdashboard, size=[0:255;])
end

function compileautumn()
  println("compileautumn")
  autumnString = postpayload(:autumnstring, "(program (= GRID_SIZE 12))")
  clientid = parse(Int64, postpayload(:clientid, 0))
  println(string("autumnstring: ", autumnString))
  println(string("clientid: ", clientid))
  println(typeof(clientid))
  if !haskey(MODS, clientid)
    try
      parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
      println("PARSED SUCCESSFULLY")
      compiledAutumn = compiletojulia(parsedAutumn)
      println("COMPILED SUCCESSFULLY")
      mod = eval(compiledAutumn)
      println("EVALUATED SUCCESSFULLY")
      content = showstring(parsedAutumn)
      MODS[clientid] = mod
      println("HERE 3")
    catch y
      println("PARSING OR COMPILING FAILURE!")
      println(y)
      content = ""
    end
  else
    println("RESET")
    content = autumnString
    delete!(MODS, clientid)
  end
  json([content])
end

function step()
  println("step")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  println(state.time)
  next_state = mod.next(state, nothing, nothing, nothing, nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  STATES[clientid] = next_state
  #HISTORY[clientid][next_state.time] = cells
  json(vcat(background, cells))
end

function startautumn()
  println("startautumn")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = mod.init(nothing, nothing, nothing, nothing, nothing)
  println(state.time)
  grid_size = state.GRID_SIZEHistory[state.time]
  background = :backgroundHistory in fieldnames(typeof(state)) ? state.backgroundHistory[state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(state.scene))
  STATES[clientid] = state
  #HISTORY[clientid] = Dict{Int, Any}()
  #HISTORY[clientid][state.time] = cells
  json(vcat(grid_size, background, cells))
  # json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].init(nothing)) : []))
end

function clicked()
  println("click")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  println(state.time)
  next_state = mod.next(state, mod.Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))), nothing, nothing, nothing, nothing)
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  STATES[clientid] = next_state
  #HISTORY[clientid][next_state.time] = cells
  json(vcat(background, cells))
  #json(map(particle -> [particle.position.x, particle.position.y, particle.color], haskey(MODS, clientid) ? filter(particle -> particle.render, MODS[clientid].next(MODS[clientid].Click(parse(Int64, @params(:x)), parse(Int64, @params(:y))))) : []))
end

function up()
  println("up")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  next_state = mod.next(state, nothing, nothing, nothing, mod.Up(), nothing)
  STATES[clientid] = next_state
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  #HISTORY[clientid][next_state.time] = cells
  json(vcat(background, cells))
end

function down()
  println("down")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  next_state = mod.next(state, nothing, nothing, nothing, nothing, mod.Down())
  STATES[clientid] = next_state
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  #HISTORY[clientid][next_state.time] = cells
  json(vcat(background, cells))
end

function right()
  println("right")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  next_state = mod.next(state, nothing, nothing, mod.Right(), nothing, nothing)
  STATES[clientid] = next_state
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  #HISTORY[clientid][next_state.time] = cells
  json(vcat(background, cells))
end

function left()
  println("left")
  clientid = parse(Int64, @params(:clientid))
  mod = MODS[clientid]
  state = STATES[clientid]
  next_state = mod.next(state, nothing, mod.Left(), nothing, nothing, nothing)
  STATES[clientid] = next_state
  background = (:backgroundHistory in fieldnames(typeof(next_state))) ? next_state.backgroundHistory[next_state.time] : "#ffffff00"
  cells = map(cell -> [cell.position.x, cell.position.y, cell.color], mod.render(next_state.scene))
  #HISTORY[clientid][next_state.time] = vcat(background, cells)
  json(vcat(background, cells))
end

function replay()
  println("replay")
  clientid = parse(Int64, @params(:clientid))
  json(HISTORY[clientid])
  # json(Dict(key => map(particle -> [particle.position.x, particle.position.y, particle.color], filter(particle -> particle.render, particles)) for (key, particles) in history))
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
    Expr(:let, vars...) => "let \n\t$(join(map(showstring, vars[1:end-1]), "\n\t"))\nin\n\t$(showstring(vars[end]))"
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
fg(s::Cons) = array(s)
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
    println(expr.args[1])
    println(expr.args[2])
    data["types"][expr.args[1]] = expr.args[2]
    :()
  else
    :(local $(compile(expr.args[1], data))::$(compile(expr.args[2], data)))
  end
end

function compileexternal(expr::AExpr, data::Dict{String, Any})
  println("here: ")
  println(expr.args[1])
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
  if !(fnName in binaryOperators) && fnName != :prev
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
  println("here")
  println(typeof(expr.args[1]) == AExpr ? expr.args[1].args[1] : expr.args[1])
  event = compile(expr.args[1], data)
  if event in [:click, :left, :right, :up, :down]
    push!(data["on"], (:(occurred($(event))), compile(expr.args[2], data)))
  elseif expr.args[1].head == :call && expr.args[1].args[1] == :clicked
    println("maybe?")
    push!(data["on"], (:(clicked(click, $(map(x -> compile(x, data), expr.args[1].args[2:end])...))), compile(expr.args[2], data)))
  else
    push!(data["on"], (event, compile(expr.args[2], data)))
  end
  :()
end

function compileinitnext(data::Dict{String, Any})
  init = quote
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), data["lifted"])...)
    $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2].args[1], data))), data["initnext"])...)
  end

  onClauses = map(x -> quote 
    if $(x[1])
      $(map(x -> :($(compile(x.args[1], data)) = state.$(Symbol(string(x.args[1])*"History"))[state.time - 1]), 
      vcat(data["initnext"], data["lifted"]))...)
      $(x[2])
    end
  end, data["on"])

  next = quote
    $(map(x -> :($(compile(x.args[1], data)) = state.$(Symbol(string(x.args[1])*"History"))[state.time - 1]), 
      vcat(data["initnext"], data["lifted"]))...)
    $(vcat(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2].args[2], data))), data["initnext"]),
           map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))
      )...)
    $(onClauses...)
  end

  initFunction = quote
    function init($(map(x -> :($(compile(x.args[1], data))::Union{$(compile(data["types"][x.args[1]], data)), Nothing}), data["external"])...))::STATE
      $(compileinitstate(data))
      $(init)
      $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] != :GRID_SIZE, data["lifted"]))...)
      $(map(x -> :(state.$(Symbol(string(x.args[1])*"History"))[state.time] = $(compile(x.args[1], data))), 
            vcat(data["external"], data["initnext"], data["lifted"]))...)
            state.scene = Scene(vcat([$(filter(x -> get(data["types"], x, :Any) in vcat(data["objects"], map(x -> [:List, x], data["objects"])), 
        map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))...)]...), :backgroundHistory in fieldnames(STATE) ? state.backgroundHistory[state.time] : "#ffffff00")
      deepcopy(state)
    end
    end
  nextFunction = quote
    function next($([:(old_state::STATE), map(x -> :($(compile(x.args[1], data))::Union{$(compile(data["types"][x.args[1]], data)), Nothing}), data["external"])...]...))::STATE
      global state = deepcopy(old_state)
      state.time = state.time + 1
      $(map(x -> :($(compile(x.args[1], data)) = $(compile(x.args[2], data))), filter(x -> x.args[1] == :GRID_SIZE, data["lifted"]))...)
      $(next)
      $(map(x -> :(state.$(Symbol(string(x.args[1])*"History"))[state.time] = $(compile(x.args[1], data))), 
            vcat(data["external"], data["initnext"], data["lifted"]))...)
      state.scene = Scene(vcat([$(filter(x -> get(data["types"], x, :Any) in vcat(data["objects"], map(x -> [:List, x], data["objects"])), 
        map(x -> x.args[1], vcat(data["initnext"], data["lifted"])))...)]...), :backgroundHistory in fieldnames(STATE) ? state.backgroundHistory[state.time] : "#ffffff00")
      deepcopy(state)
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
                          println(filter(cell -> !isWithinBounds(cell.position),render(obj)))
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
                          println("LOOK AT ME")
                          println(reduce(&, map(obj -> clicked(click, obj), objects)))
                          reduce(|, map(obj -> clicked(click, obj), objects))
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

                        function addObj(list::Array{<:Object}, obj::Object)
                          push!(list, obj)
                          list
                        end

                        function addObj(list::Array{<:Object}, objs::Array{<:Object})
                          list = vcat(list, objs)
                          list
                        end

                        function removeObj(list::Array{<:Object}, obj::Object)
                          old_obj = filter(x -> x.id == obj.id, list)
                          old_obj.alive = false
                          list
                        end

                        function removeObj(list::Array{<:Object}, fn)
                          orig_list = filter(obj -> !fn(obj), list)
                          removed_list = filter(obj -> fn(obj), list)
                          foreach(obj -> (obj.alive = false), removed_list)
                          vcat(orig_list, removed_list)
                        end

                        function removeObj(obj::Object)
                          obj.alive = false
                          obj
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

                        function isWithinBounds(position::Position)::Bool
                          (position.x >= 0) && (position.x < state.GRID_SIZEHistory[0]) && (position.y >= 0) && (position.y < state.GRID_SIZEHistory[0])                          
                        end

                        function isFree(position::Position)::Bool
                          length(filter(cell -> cell.position.x == position.x && cell.position.y == position.y, render(state.scene))) == 0
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

                        function adjacent(cell1::Cell, cell2::Cell)
                          adjacent(cell1.position, cell2.position)
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

                        function moveNoCollision(object::Object, position::Position)::Object
                          (isWithinBounds(move(object, position)) && isFree(move(object, position.x, position.y), object)) ? move(object, position.x, position.y) : object 
                        end

                        function moveNoCollision(object::Object, x::Int, y::Int)
                          (isWithinBounds(move(object, position)) && isFree(move(object, x, y), object)) ? move(object, x, y) : object 
                        end

                        function randomPositions(GRID_SIZE::Int, n::Int)::Array{Position}
                          nums = uniformChoice([0:(GRID_SIZE * GRID_SIZE - 1);], n)
                          println(nums)
                          println(map(num -> Position(num % GRID_SIZE, floor(Int, num / GRID_SIZE)), nums))
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
                          println("nextLiquid")
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
                          println("nextSolid")
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

end