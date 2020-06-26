module AutumnModelsController
using Genie.Renderer
using Genie.Renderer.Html: html
using Genie.Renderer.Json: json
using Genie.Router
using MLStyle
using SExpressions

GRID_SIZE = 16;
FORM_CONTENT = "";
MOD = nothing
CLICK = nothing;
PARTICLES = []
ERROR = ""
compiled = false

function autumnmodels()
  redirect(:playground)
end

function playground()
  html(:autumnmodels, :autumnmodelsdashboard, 
  size=[0:GRID_SIZE*GRID_SIZE-1;], 
  content=FORM_CONTENT, 
  particles=map(particle -> GRID_SIZE*(particle.position.y) + particle.position.x, PARTICLES),
  compiled=compiled, GRID_SIZE=GRID_SIZE, challenges=false)
end

function challenges()
  html(:autumnmodels, :autumnmodelsdashboard, 
  size=[0:GRID_SIZE*GRID_SIZE-1;], 
  content=FORM_CONTENT, 
  particles=map(particle -> GRID_SIZE*(particle.position.y) + particle.position.x, PARTICLES),
  compiled=compiled, GRID_SIZE=GRID_SIZE, challenges=true)
end

function compileautumn()
  autumnString = @params(:autumnstring)
  if !compiled
    try
      parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
      println("PARSED SUCCESSFULLY")
      compiledAutumn = compiletojulia(parsedAutumn)
      println("COMPILED SUCCESSFULLY")
      global MOD = eval(compiledAutumn)
      println("EVALUATED SUCCESSFULLY")
      global compiled = true
      global GRID_SIZE = MOD.GRID_SIZE
      global FORM_CONTENT = showstring(parsedAutumn)
      println("HERE 3")
    catch y
      println("PARSING OR COMPILING FAILURE!")
      println(y)
      global FORM_CONTENT = "" 
      global GRID_SIZE = 16
      global MOD = nothing
    end
  else
    global GRID_SIZE = 16
    global FORM_CONTENT = autumnString
    global MOD = nothing
    global CLICK = nothing
    global PARTICLES = []
    global compiled = false
  end
  redirect(:get)
end

function step()
  println("step")
  json(map(particle -> [particle.position.x, particle.position.y, particle.color], MOD !== nothing ? MOD.next(nothing) : []))
end

function startautumn()
  println("startautumn")
  
  json(map(particle -> [particle.position.x, particle.position.y, particle.color], MOD !== nothing ? MOD.init(nothing) : []))
end

function clicked()
  println("clicked params:")
  println(@params(:x))
  println(@params(:y))
  
  json(map(particle -> [particle.position.x, particle.position.y, particle.color], MOD !== nothing ? MOD.next(MOD.Click(parse(Int64, @params(:x)), parse(Int64, @params(:y)))) : []))

end

# aexpr.jl
const autumngrammar = """
x           := a | b | ... | aa ...
program     := statement*
statement   := externaldecl | assignexpr | typedecl | typedef

typedef     := type fields  #FIXME
typealias   := "type alias" type fields
fields      := field | fields field
field       := constructor | constructor typesymbol*
cosntructor := typesymbol

typedecl    := x : typeexpr
externaldecl:= external typedecl

assignexpr  := x = valueexpr

typeexpr    := typesymbol | paramtype | typevar | functiontype
funtype     := typeexpr -> typeexpr
producttype := typeexpr × typexexpr × ...
typesymbol  := primtype | customtype
primtype    := Int | Bool | Float
customtype  := A | B | ... | Aa | ...

valueexpr   := fappexpr | lambdaexpr | iteexpr | initnextexpr | letexpr |
               this | lambdaexpr
iteexpr     := if expr then expr else expr
intextexpr  := init expr next expr
fappexpr    := valueexpr valueexpr*
letexpr     := let x = valueexpr in valueexpr
lambdaexpr  := x --> expr
"""

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

# wrap(expr::Expr) = AExpr(expr)
# wrap(x) = x

# AExpr(xs...) = AExpr(Expr(xs...))

# function Base.getproperty(aexpr::AExpr, name::Symbol)
#   expr = getfield(aexpr, :expr)
#   if name == :expr
#     expr
#   elseif name == :head
#     expr.head
#   elseif name == :args
#     expr.args
#   else
#     error("no property $name of AExpr")
#   end
# end


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
    Expr(:if, i, t, e) => "if ($(showstring(i))) then ($(showstring(t))) else ($(showstring(e)))"
    Expr(:initnext, i, n) => "init $(showstring(i)) next $(showstring(n))"
    Expr(:args, args...) => join(map(showstring, args), " ")
    Expr(:call, f, arg1, arg2) && if isinfix(f) end => "$(showstring(arg1)) $f $(showstring(arg2))"
    Expr(:call, f, args...) => "($(join(map(showstring, [f ; args]), " ")))"
    Expr(:let, vars, val) => "let \n\t$(join(map(showstring, vars), "\n\t"))\n in \n\t $(showstring(val))"
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
    Expr(:const, assignment) => "const $(showstring(assignment))"
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
    [:const, assignment]              => AExpr(:const, parseau(assignment))
    [:let, vars, val]                 => AExpr(:let, map(parseau, vars), parseau(val))
    [:case, name, cases...]           => AExpr(:case, name, map(parseau, cases)...)
    [:(=>), type, value]              => AExpr(:casevalue, parseau(type), parseau(value))
    [:type, :alias, var, val]         => AExpr(:typealias, var, parsealias(val))
    [:fn, params, body]               => AExpr(:fn, AExpr(:list, params...), parseau(body))
    [:(-->), var, val]                => AExpr(:lambda, parseau(var), parseau(val))
    [:list, vars...]                  => AExpr(:list, map(parseau, vars)...)
    [:.., var, field]                 => AExpr(:field, parseau(var), parseau(field))
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

parseau(list::Array{BigInt, 1}) = list[1]
parsetypeau(s::Symbol) = s
parseau(s::Symbol) = s
parseau(s::Union{Number, String}) = s

macro au_str(x::String)
  QuoteNode(parseautumn(x))
end

# compile.jl

binaryOperators = [:+, :-, :/, :*, :&, :||, :>=, :<=, :>, :<, :(==), :!=]

struct AutumnCompileError <: Exception
  msg
end
AutumnCompileError() = AutumnCompileError("")

"compile `aexpr` into Expr"
function compiletojulia(aexpr::AExpr)::Expr

  data = Dict([("historyVars" => []),
               ("externalVars" => []),
               ("initnextVars" => []),
               ("liftedVars" => []),
               ("types" => Dict())])
  
  # ----- HELPER FUNCTIONS ----- #
  function compile(expr::AExpr, parent=nothing)
    arr = [expr.head, expr.args...]
    res = @match arr begin
      [:if, args...] => :($(compile(args[1], expr)) ? $(compile(args[2], expr)) : $(compile(args[3], expr)))
      [:assign, args...] => compileassign(expr, parent, data)
      [:typedecl, args...] => compiletypedecl(expr, parent, data)
      [:external, args...] => compileexternal(expr, data)
      [:const, args...] => :(const $(compile(args[1].args[1])) = $(compile(args[1].args[2])))
      [:let, args...] => compilelet(expr)
      [:case, args...] => compilecase(expr)
      [:typealias, args...] => compiletypealias(expr)
      [:lambda, args...] => :($(compile(args[1])) -> $(compile(args[2])))
      [:list, args...] => :([$(map(compile, expr.args)...)])
      [:call, args...] => compilecall(expr)
      [:field, args...] => :($(compile(expr.args[1])).$(compile(expr.args[2])))
      [args...] => throw(AutumnCompileError("Invalid AExpr Head: "))
    end
  end

  function compile(expr::AbstractArray, parent=nothing)
    if length(expr) == 0 || (length(expr) > 1 && expr[1] != :List)
      throw(AutumnCompileError("Invalid Compound Type"))
    elseif expr[1] == :List
      :(Array{$(compile(expr[2:end]))})
    else
      expr[1]      
    end
  end

  function compile(expr, parent=nothing)
    expr
  end
  
  function compileassign(expr::AExpr, parent::AExpr, data::Dict{String, Any})
    # get type, if declared
    type = haskey(data["types"], (expr.args[1], parent)) ? data["types"][(expr.args[1], parent)] : nothing
    if (typeof(expr.args[2]) == AExpr && expr.args[2].head == :fn)
      if type !== nothing # handle function with typed arguments/return type
        args = compile(expr.args[2].args[1]).args # function args
        argTypes = map(compile, type.args[1:(end-1)]) # function arg types
        tuples = [(arg, type) for arg in args, type in argTypes]
        typedArgExprs = map(x -> :($(x[1])::$(x[2])), tuples)
        quote 
          function $(compile(expr.args[1]))($(typedArgExprs...))::$(compile(type.args[end]))
            $(compile(expr.args[2].args[2]))  
          end
        end 
      else # handle function without typed arguments/return type
        quote 
          function $(compile(expr.args[1]))($(compile(expr.args[2].args[1]).args[2]...))
              $(compile(expr.args[2].args[2]))  
          end 
        end          
      end
    else # handle non-function assignments
      # handle global assignments
      if parent !== nothing && (parent.head == :program) 
        push!(data["historyVars"], (expr.args[1], parent))
        if (typeof(expr.args[2]) == AExpr && expr.args[2].head == :initnext)
          push!(data["initnextVars"], expr)
        elseif (parent.head == :program)
          push!(data["liftedVars"], expr)
        end
        :()
      # handle non-global assignments
      else 
        if type !== nothing
          :($(compile(expr.args[1]))::$(compile(type)) = compile(expr.args[2]))
        else
            :($(compile(expr.args[1])) = $(compile(expr.args[2])))
        end
      end
    end
  end
  
  function compiletypedecl(expr, parent, data)
    data["types"][(expr.args[1], parent)] = expr.args[2]
    :()
  end
  
  function compileexternal(expr, data)
    push!(data["externalVars"], expr.args[1].args[1])
    push!(data["historyVars"], (expr.args[1].args[1], expr))
    compiletypedecl(expr.args[1], expr, data)
  end
  
  function compiletypealias(expr)
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
  
  function compilecall(expr)
    fnName = expr.args[1]
      if !(fnName in binaryOperators) && fnName != :prev
        :($(fnName)($(map(compile, expr.args[2:end])...)))
      elseif fnName == :prev
        :($(Symbol(string(expr.args[2]) * "Prev"))($(map(compile, expr.args[3:end])...)))
      elseif fnName != :(==)        
        :($(fnName)($(compile(expr.args[2])), $(compile(expr.args[3]))))
      else
        :($(compile(expr.args[2])) == $(compile(expr.args[3])))
      end
  end
  
  function compilelet(expr)
    quote
      $(vcat(map(x -> compile(x, expr), expr.args[1]), compile(expr.args[2]))...)
    end
  end

  function compilecase(expr)
    quote 
      @match $(compile(expr.args[1])) begin
        $(map(x -> :($(compile(x.args[1])) => $(compile(x.args[2]))), expr.args[2:end])...)
      end
    end
  end
  
  function compileinitnext(data)
    initFunction = quote
      function init($(map(x -> compile(x), data["externalVars"])...))
        $(map(x -> :(global $(compile(x.args[1])) = $(compile(x.args[2].args[1]))), data["initnextVars"])...)
        $(map(x -> :(global $(compile(x.args[1])) = $(compile(x.args[2]))), data["liftedVars"])...)
        $(map(x -> :($(Symbol(string(x[1])*"History"))[time] = deepcopy($(compile(x[1])))), data["historyVars"])...)
        particles
      end
     end
    nextFunction = quote
      function next($(map(x -> compile(x), data["externalVars"])...))
        global time += 1
        $(map(x -> :(global $(compile(x.args[1])) = $(compile(x.args[2].args[2]))), data["initnextVars"])...)
        $(map(x -> :(global $(compile(x.args[1])) = $(compile(x.args[2]))), data["liftedVars"])...)
        $(map(x -> :($(Symbol(string(x[1]) * "History"))[time] = deepcopy($(compile(x[1])))), data["historyVars"])...)
        particles
      end
     end
     [initFunction, nextFunction]
  end
  # ----- END HELPER FUNCTIONS ----- #


  # ----- COMPILATION -----#
  if (aexpr.head == :program)
    # handle AExpression lines
    lines = filter(x -> x !== :(),map(arg -> compile(arg, aexpr), aexpr.args))
    
    # handle history 
    initGlobalVars = map(expr -> :($(compile(expr[1])) = nothing), data["historyVars"])
    push!(initGlobalVars, :(time = 0))
    # non-external variable history dicts
    initHistoryDictArgs = map(expr -> 
      :($(Symbol(string(expr[1]) * "History")) = Dict{Int64, $(haskey(data["types"], expr) ? compile(data["types"][expr]) : Any)}()),
      filter(x -> !(x[1] in data["externalVars"]), data["historyVars"]))
    # external variable history dicts
    initHistoryDictArgs = vcat(initHistoryDictArgs, map(expr -> 
    :($(Symbol(string(expr[1]) * "History")) = Dict{Int64, Union{$(compile(data["types"][expr])), Nothing}}())
    , filter(x -> (x[1] in data["externalVars"]), data["historyVars"])))
    
    # handle initnext
    initnextFunctions = compileinitnext(data)
    prevFunctions = compileprevfuncs(data)
    builtinFunctions = compilebuiltin()

    # remove empty lines
    lines = filter(x -> x != :(), 
            vcat(builtinFunctions, initGlobalVars, lines, initHistoryDictArgs, prevFunctions, initnextFunctions))

    # construct module
    expr = quote
      module CompiledProgram
        export init, next
        using Distributions
        using MLStyle: @match 
        $(lines...)
      end
    end  
    expr.head = :toplevel
    expr
  else
    throw(AutumnCompileError())
  end
end

"Run `prog` forever"
function runprogram(prog::Expr, n::Int)
  mod = eval(prog)
  particles = mod.init(mod.Click(5, 5))

  externals = [nothing, nothing]
  for i in 1:n
    externals = [nothing, mod.Click(rand([1:10;]), rand([1:10;]))]
    particles = mod.next(mod.next(externals[rand(Categorical([0.7, 0.3]))]))
  end
  particles
end

# ----- Built-In and Prev Function Helpers ----- #

function compileprevfuncs(data::Dict{String, Any})
  prevFunctions = map(x -> quote
        function $(Symbol(string(x[1]) * "Prev"))(n::Int=1)
          $(Symbol(string(x[1]) * "History"))[time - n] 
        end
        end, 
  data["historyVars"])
  prevFunctions
end

function compilebuiltin()
  occurredFunction = builtInDict["occurred"]
  uniformChoiceFunction = builtInDict["uniformChoice"]
  clickType = builtInDict["clickType"]
  [occurredFunction, uniformChoiceFunction, clickType]
end

builtInDict = Dict([
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
"clickType"       =>  quote
                        struct Click
                          x::Int
                          y::Int                    
                        end     
                      end
])




end