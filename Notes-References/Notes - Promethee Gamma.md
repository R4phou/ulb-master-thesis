[[Promethee Gamma.pdf]]
Method for partial #ranking based on valued coalitions of monocriterion net flow scores.

#MCDA divided in 3 parts:
- Aggregating - complete ranking of the set of alternatives
- Interactive - 
- Outranking methods - allow some pairs of alternatives to remain incomparable
	- If strong conflicting information in the two alternatives
	- Not enough elements in the decision problem to state the preference/indifference between them
	- #PROMETHEE 

This paper **provides** 
- an analysis of the incomparability relation produced by #PROMETHEE1.
- Propose a new method #PROMETHEEGAMMA based on the comparison of weighted coalitions of mono-criterion net flow score differences.
- Comparison of the 2.


## PROMETHEE 1
#PROMETHEE1 property:
![](Pasted%20image%2020241003172833.png)

#PROMETHEE1 has 3 **limitations** (see paper):
1. #Usual-Criterion
2. Particular cases of unexpected incomparability relations
3. Increasing indifference $q$ threshold produces incomparability

To encounter this #PROMETHEEGAMMA has been proposed.
### Usual criterion
The #Usual-Criterion is a problem characterized by the **ordinal scales**. Leading to meaningless computation of differences of evaluations.
	An **ordinal scale** is a type of measurement scale that involves ordering or ranking elements based on a specific characteristic, but without establishing the degree of difference between the ranks. In other words, an ordinal scale allows for the _ordering_ of data points, but the intervals between these points are not meaningful or consistent.

The only acceptable preference function is the usual one:![](Pasted%20image%2020241105131332.png)
We **expect thus to get at least as much incomparability** relations as with other types of preference functions (since strong monocriterion conflicts).

**BUT**, it is *not possible for two alternatives to be considered incomparable* according to the #PROMETHEE1 method hen using the #usual-criterion if all alternatives have different evaluations on any criterion.
	Since we *suppose that all alternatives have different evaluations*, we can compute the negative outranking flow score of any alternative $a_i$ with the positive outranking flow 
		![](Pasted%20image%2020241105132249.png)
	And therefore we deduce that:![](Pasted%20image%2020241105132331.png)

In this setting, **two pairs of alternatives can never be considered as incomparable according to PROMETHEE I**. However, in PROMETHEE I , the use of ordinal criteria (with all alternatives having different evaluations) leads to the full compensations of the advantages and drawbacks on the different criteria.
### Particular cases of unexpected incomparability relations
[Promethee Gamma](Promethee%20Gamma.pdf#page=5)
#Rank-reversal #RR can happen only between pairs of alternatives having small differences of net flow scores and show that the net flow score procedure is respecting a strict form of monotonicity .

$\implies$ Indifference relation in the context of PROMETHEE methods should be reconsidered
### Increasing indifference threshold produces incomparability
Considering a **linear preference function**.

**Increasing** $q_{c}$ can for each pair of alternatives $a_{i}$ and $a_{j}$ **only decrease** the pairwise preferences $\pi_{ix}^{c}$ and $\pi_{xj}^{c}$ for any $a_{x}\in A$ and similarly **only increase** $\pi_{jx}^{c}$ and $\pi_{xi}^{c}$ for any $a_{x}\in A$.

Which should result only in the decrease of the arguments stating that $a_{i}$ is preferred to $a_{j}$ or that $a_{j}$ is preferred to $a_{i}$ respectively.

**However**, this is not always the case as shown here:
[Promethee Gamma](Promethee%20Gamma.pdf#page=6)

## Promethee Gamma
PROMETHEE methods suffers from the #RR phenomenon:
- When one alternative is removed or added to the dataset
- The respective order in the ranking of two other alternatives can be reversed

Even though #PROMETHEE2 should **not be applied** for **decision problem**, there exists a solid foundation on the computation of the net flow scores.

For these reasons a new variant is created #PROMETHEEGAMMA, it extends the notion of **net flow scores** in order to *model incomparabilities*.
- It works by computing for each pair of alternatives, some **aggregated pairwise preference indicators** $\gamma_{ij}$ and $\gamma_{ji}$.
- These indicators will *not only depend on the mutual pairwise comparison of the concerned alternatives*, but also on **how they behave pairwise** with all the other alternatives of the problem.
- It will be shown that these indicators reflect the net flow scores as the difference between  $\gamma_{ij}$ and $\gamma_{ji}$ is equal to their difference of net flow score.
- Downside is: the decision maker needs to **select 3 at most additional preference parameters**
### PROMETHEE GAMMA Presentation
![](Pasted%20image%2020241003174451.png)
with $\phi^{c}(a_{i})$ from #PROMETHEE2 

$\gamma_{ij}$ represents the global advantages of $a_{i}$ over $a_{j}$ in the **whole data set**.
- Incuded in $[0,2]$ interval
- If $w_{c}=cst\implies \gamma_{ij}$ can be seen as the su, of the number of alternatives having their evaluations between ai and aj for all criteria where ai is better than aj
- Interpretation of difference between these two indicators as a difference of #Borda scores for the net flow score.
- $\gamma_{ij}-\gamma_{ji}=\phi(a_{i})- \phi(a_{j})$

It is the importance of the coalition of weights for which $a_{i}$ is better than $a_{j}$ with the weights being themselves weighed by the difference of the monocriterion net flow scores.

![](Pasted%20image%2020241003175202.png)
- TI -  global indifference threshold
- TJ -  a global incomparability threshold 
- Pf - and a global preference factor
To be chosen by decision maker

#todo P7 bas a gauche