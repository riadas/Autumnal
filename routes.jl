using Genie.Router
using AutumnModelsController

route("/", AutumnModelsController.autumnmodels)

route("/compileautumn", AutumnModelsController.compileautumn, method = POST, named=:compile_autumn)

route("/startautumn", AutumnModelsController.startautumn, method = GET, named=:startautumn)

route("/clicked", AutumnModelsController.clicked, method = GET, named=:clicked)

route("/step", AutumnModelsController.step, method = GET, named=:step)