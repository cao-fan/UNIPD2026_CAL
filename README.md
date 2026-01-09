Install Julia 1.11.7

Open the terminal and cd to the installation folder and open the Julia REPL:

> julia

or if multiple versions of julia are installed:

> julia +1.11.7

Enter the Package Manager mode by pressing ']', then run:

> activate .

Then run 'instantiate' to install all the requirements:

> instantiate

Backslash to exit Package Manager mode and then run:

> include("script.jl")
