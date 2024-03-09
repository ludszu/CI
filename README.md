# Configuration interaction (CI) method for a gated MoS2 quantum dot (QD)

## Interacting problem

We are solving an interacting problem of N electrons on M single particle (SP) states in a single gated MoS$_2$ quantum dot. The interacting Hamiltonian is

$H=\sum_i \epsilon_i c_i^{\dagger}c_i+\frac{1}{2}\sum_{ijkl}V_{ijkl}c_i^{\dagger}c_j^{\dagger}c_kc_l$,

where $i,j,k,l$ label SP states and $epsilon_i, V_{ijkl}$ are SP eigenenergies and Coulomb matrix elements (CME) that need to be calculated prior to a CI calculation.

The solution is a linear combination of all possible configurations of N electrons on M SP orbitals, e.g. for 3 electrons we have

$\Psi_(N=3)=\sum_{pqs}A_{pqs}c_r^{dagger}c_q^{dagger}c_p^{dagger}|0>=\sum_{pqs}A_{pqs}|pqs>$,

where $p,q,s$ label SP states and $|pqs>$ is a single Slater determinant. We obtain $A_{pqs}$ through diagonalising the Hamiltonian.


## Input files

The input files consist of
* SP energy levels for spin up (U) and down (D)
* CME elements for spin combinations UU, DD, UD for bare Coulomb interaction and Keldysh screening
* quantum numbers specific to the problem of MoS2 QD: valley index and angular momentum

  
