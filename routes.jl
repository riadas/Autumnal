using Genie.Router
using AutumnModelsController

route("/", AutumnModelsController.autumnmodels)

route("/playground", AutumnModelsController.playground, method = GET, named=:playground)

route("/compileautumn", AutumnModelsController.compileautumn, method = POST, named=:compile_autumn)

route("/startautumn", AutumnModelsController.startautumn, method = GET, named=:startautumn)

route("/clicked", AutumnModelsController.clicked, method = GET, named=:clicked)

route("/left", AutumnModelsController.left, method = GET, named=:left)

route("/right", AutumnModelsController.right, method = GET, named=:right)

route("/up", AutumnModelsController.up, method = GET, named=:up)

route("/down", AutumnModelsController.down, method = GET, named=:down)

route("/step", AutumnModelsController.step, method = GET, named=:step)

route("/replay", AutumnModelsController.replay, method = GET, named=:replay)