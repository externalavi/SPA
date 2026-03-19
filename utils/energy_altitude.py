
import random

# expected energy
# expected energy between two nodes u,v can be changing according to scenario
# considering the range of energy between node u,v is between 10 to 40


def ExpectedEnergy(u, v):
    return random.uniform(10, 40)


# altitude gain between node u and v
# altitude can vary
# from very high to very low
# for this ablation study taking the range of altitude gain between node u, v between -2 to 5
def AltitudeGain(u, v):
    return random.uniform(-2, 5)
