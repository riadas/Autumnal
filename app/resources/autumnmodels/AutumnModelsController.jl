module AutumnModelsController
using Genie.Renderer.Html
using Genie.Router
using CausalDiscovery

GRID_SIZE = 16;
FORM_CONTENT = "";

function autumnmodels()
  println(length(FORM_CONTENT))
  html(:autumnmodels, :autumnmodelsdashboard, size=[1:GRID_SIZE*GRID_SIZE;], content=FORM_CONTENT)
end

function compileautumn()
  autumnString = @params(:autumnstring)
  if autumnString != ""
    parsedAutumn = eval(Meta.parse("au\"\"\"$(autumnString)\"\"\""))
    compiledAutumn = compiletojulia(parsedAutumn)
    mod = eval(compiledAutumn)

    global GRID_SIZE = mod.GRID_SIZE
    global FORM_CONTENT = showstring(parsedAutumn)
  else
    global GRID_SIZE = 16
    global FORM_CONTENT = autumnString
  end
  autumnmodels()
end

end