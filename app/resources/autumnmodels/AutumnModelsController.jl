module AutumnModelsController
using Genie.Renderer
using Genie.Renderer.Html: html
using Genie.Router
using Base
using CausalDiscovery

GRID_SIZE = 16;
FORM_CONTENT = "";
MOD = nothing
CLICK = nothing;
PARTICLES = []
ERROR = ""
running = false

function autumnmodels()
  html(:autumnmodels, :autumnmodelsdashboard, 
       size=[0:GRID_SIZE*GRID_SIZE-1;], 
       content=FORM_CONTENT, 
       particles=map(particle -> GRID_SIZE*(particle.position.y) + particle.position.x, PARTICLES),
       running=running)
end

function compileautumn()
  autumnString = @params(:autumnstring)
  if autumnString != ""
      parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
      compiledAutumn = compiletojulia(parsedAutumn)
      global MOD = eval(compiledAutumn)

      global GRID_SIZE = MOD.GRID_SIZE
      global FORM_CONTENT = showstring(parsedAutumn)
  else
    global GRID_SIZE = 16
    global FORM_CONTENT = autumnString
    global MOD = nothing
    global CLICK = nothing
    global PARTICLES = []
    global running = false
  end
  redirect(:get)
end

function step()
  if (MOD !== nothing)
    global PARTICLES = MOD.next(CLICK)
  end
  global CLICK = nothing
  redirect(:get)
end

function runautumn()
  if (MOD !== nothing)
    MOD.init(nothing)
  end
  global running = true
  redirect(:get)
end

function stopautumn()
  println("stopautumn")
  global running = false
  
  redirect(:get)
end

function clicked()
  println("clicked params:")
  id = parse(Int, @params(:id)[2:(end-1)])
  println(id)
  global CLICK = MOD.Click(id % GRID_SIZE, floor(Int, id/GRID_SIZE))
  redirect(:get)
end

end