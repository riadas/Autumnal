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
compiled = false
running = false
not_run_yet = true

function autumnmodels()
  html(:autumnmodels, :autumnmodelsdashboard, 
       size=[0:GRID_SIZE*GRID_SIZE-1;], 
       content=FORM_CONTENT, 
       particles=map(particle -> GRID_SIZE*(particle.position.y) + particle.position.x, PARTICLES),
       running=running, compiled=compiled, not_run_yet=not_run_yet)
end

function compileautumn()
  autumnString = @params(:autumnstring)
  if autumnString != ""
    try
      parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
      compiledAutumn = compiletojulia(parsedAutumn)
      global MOD = eval(compiledAutumn)
      global compiled = true
      global GRID_SIZE = MOD.GRID_SIZE
      global FORM_CONTENT = showstring(parsedAutumn)
    catch ErrorException
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
    global running = false
    global compiled = false
    global not_run_yet = true;
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
  global not_run_yet = false;
  redirect(:get)
end

function stopautumn()
  println("stopautumn")
  global running = !running;
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