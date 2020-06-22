module AutumnModelsController
using Genie.Renderer.Html

GRID_SIZE = 16;

function autumnmodels()
  html(:autumnmodels, :autumnmodelsdashboard, size=[1:GRID_SIZE*GRID_SIZE;])
end

end