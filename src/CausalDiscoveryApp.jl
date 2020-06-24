module CausalDiscoveryApp

using Logging

function main()
  Base.eval(Main, :(const UserApp = CausalDiscoveryApp))

  include(joinpath("..", "genie.jl"))

  Base.eval(Main, :(const Genie = CausalDiscoveryApp.Genie))
  Base.eval(Main, :(using Genie))
end; main()

end
