import math
import random
import os


# Synapse
class Synapse(object):
    def __init__(self):
        self.inpuT = 0
        self.output = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0

    def clone(self):
        synapse = Synapse()
        synapse.inpuT = self.inpuT
        synapse.output = self.output
        synapse.weight = self.weight
        synapse.enabled = self.enabled
        synapse.innovation = self.innovation
        return synapse

    def printSynapse(self):
        print("input:", self.inpuT)
        print("output:", self.output)
        print("weight:", self.weight)
        print("enabled:", self.enabled)
        print("innvation:", self.innovation)


# Neuron
class Neuron(object):
    def __init__(self):
        self.value = 0.0
        self.inputs = []

    @staticmethod
    def sigmoid(x):

        y = 0

        try:
            y = 1.0 / (1.0 + math.exp(-4.9 * x))
        except OverflowError:
            pass
        return y

    def printNeuron(self):
        print("value:", self.value)
        if len(self.inputs) == 0:
            print("no synapse in the input list")
        else:
            print("INPUT LIST")
            for synapse in self.inputs:
                print("SYNAPSE:")
                synapse.printSynapse()
        print()


# Species


class Species(object):
    CROSSOVER = 0.75

    def __init__(self):
        self.genomes = []  # ArrayList[Genome]()
        self.topFitness = 0.0
        self.averageFitness = 0.0
        self.staleness = 0

    def printSpecies(self):
        print("topFitness:", self.topFitness)
        print("averageFitness:", self.averageFitness)
        print("staleness:", self.staleness)
        if len(self.genomes) == 0:
            print("no genome in the genome list")
        else:
            print("GENOME LIST")
            for genome in self.genomes:
                print("GENOME:")
                # genome.printGenome()
                print(genome)
        print()

    def breedChild(self):
        if random.random() < Species.CROSSOVER:
            g1 = self.genomes[random.randint(0, len(self.genomes) - 1)]
            g2 = self.genomes[random.randint(0, len(self.genomes) - 1)]
            child = self.crossover(g1, g2)
        else:
            child = self.genomes[random.randint(0, len(self.genomes) - 1)].clone()
        child.mutate()
        return child

    def calculateAverageFitness(self):
        total = 0.0
        for genome in self.genomes:
            total += genome.globalRank
        self.averageFitness = total / len(self.genomes)

    def crossover(self, g1, g2):
        if g2.fitness > g1.fitness:
            g1, g2 = g2, g1

        child = Genome()

        for gene1 in g1.genes:
            for gene2 in g2.genes:

                if gene1.innovation == gene2.innovation:
                    if random.choice([True, False]) and gene2.enabled:
                        child.genes.append(gene2.clone())
                        # break
                        # gotoOuterLoopFlag = True
                        break
                    else:
                        break
            child.genes.append(gene1.clone())

        child.maxNeuron = max(g1.maxNeuron, g2.maxNeuron)

        for i in range(7):
            child.mutationRates[i] = g1.mutationRates[i]

        return child


# Pool

class Pool(object):
    POPULATION = 50
    STALE_SPECIES = 15
    INPUTS = 4
    OUTPUTS = 1
    TIMEOUT = 20

    DELTA_DISJOINT = 2.0
    DELTA_WEIGHTS = 0.4
    DELTA_THRESHOLD = 1.0

    CONN_MUTATION = 0.25
    LINK_MUTATION = 2.0
    BIAS_MUTATION = 0.4
    NODE_MUTATION = 0.5
    ENABLE_MUTATION = 0.2
    DISABLE_MUTATION = 0.4
    STEP_SIZE = 0.1
    PERTURBATION = 0.9
    CROSSOVER = 0.75

    species = []
    generation = 0
    innovation = OUTPUTS
    maxFitness = 0.0

    @staticmethod
    def addToSpecies(child):
        for species in Pool.species:
            if child.sameSpecies(species.genomes[0]):
                species.genomes.append(child)
                return

        childSpecies = Species()
        childSpecies.genomes.append(child)
        Pool.species.append(childSpecies)

    @staticmethod
    def cullSpecies(cutToOne):
        for species in Pool.species:
            def getKey(custom):
                return custom.fitness

            species.genomes = sorted(species.genomes, key=getKey, reverse=True)

            remaining = math.ceil(len(species.genomes) / 2.0)
            if cutToOne:
                remaining = 1.0

            while len(species.genomes) > remaining:
                species.genomes.pop()

    @staticmethod
    def initializePool():
        for i in range(Pool.POPULATION):
            basic = Genome()
            basic.maxNeuron = Pool.INPUTS
            basic.mutate()
            Pool.addToSpecies(basic)

    @staticmethod
    def newGeneration():
        Pool.cullSpecies(False)
        Pool.rankGlobally()
        Pool.removeStaleSpecies()
        Pool.rankGlobally()

        for species in Pool.species:
            species.calculateAverageFitness()

        Pool.removeWeakSpecies()
        sumF = Pool.totalAverageFitness()
        children = []

        for species in Pool.species:
            breed = math.floor(species.averageFitness / sumF * Pool.POPULATION) - 1.0
            for i in range(int(breed)):
                children.append(species.breedChild())

        Pool.cullSpecies(True)

        while len(children) + len(Pool.species) < Pool.POPULATION:
            species = Pool.species[random.randint(0, len(Pool.species) - 1)]
            children.append(species.breedChild())

        for child in children:
            Pool.addToSpecies(child)

        Pool.generation += 1

    @staticmethod
    def rankGlobally():
        globalGenomes = []
        for species in Pool.species:
            for genome in species.genomes:
                globalGenomes.append(genome)

        def getKey(custom):
            return custom.fitness

        globalGenomes = sorted(globalGenomes, key=getKey, reverse=False)

        for i in range(len(globalGenomes)):
            globalGenomes[i].globalRank = i

    @staticmethod
    def removeStaleSpecies():
        survived = []
        for species in Pool.species:
            def getKey(custom):
                return custom.fitness

            species.genomes = sorted(species.genomes, key=getKey, reverse=True)

            if species.genomes[0].fitness > species.topFitness:
                species.topFitness = species.genomes[0].fitness
                species.staleness = 0
            else:
                species.staleness += 1

            if species.staleness < Pool.STALE_SPECIES or species.topFitness >= Pool.maxFitness:
                survived.append(species)

        Pool.species.clear()
        Pool.species.extend(survived)

    @staticmethod
    def removeWeakSpecies():
        survived = []
        sumF = Pool.totalAverageFitness()
        for species in Pool.species:
            breed = math.floor(species.averageFitness / sumF * Pool.POPULATION)
            if breed >= 1.0:
                survived.append(species)

        Pool.species.clear()
        Pool.species.extend(survived)

    @staticmethod
    def totalAverageFitness():
        total = 0
        for species in Pool.species:
            total += species.averageFitness
        return total


# Genome

class Genome(object):
    def __init__(self):
        self.genes = []  # ArrayList[Synapses]()
        self.fitness = 0.0
        self.maxNeuron = 0
        self.globalRank = 0
        self.mutationRates = [Pool.CONN_MUTATION, Pool.LINK_MUTATION, Pool.BIAS_MUTATION, \
                              Pool.NODE_MUTATION, Pool.ENABLE_MUTATION, Pool.DISABLE_MUTATION, Pool.STEP_SIZE]
        self.network = {}  # Map<Integer, Neuron> network

    def printGenome(self):
        print("genes: ", self.genes)
        print("fitness: ", self.fitness)
        print("maxNeuron: ", self.maxNeuron)
        print("globalRank: ", self.globalRank)

        print("mutationRates: ", self.mutationRates)
        print("network: ", self.network)
        print()

    def clone(self):
        genome = Genome()
        for gene in self.genes:
            genome.genes.append(gene.clone())

        genome.maxNeuron = self.maxNeuron
        for i in range(7):
            genome.mutationRates[i] = self.mutationRates[i]
        return genome

    def containsLink(self, link):
        for gene in self.genes:
            if gene.inpuT == link.inpuT and gene.output == link.output:
                return True
        return False

    def disjoint(self, genome):
        disjointGenes = 0.0
        for gene in self.genes:
            for otherGene in genome.genes:
                if gene.innovation == otherGene.innovation:
                    break
            disjointGenes += 1.0

        return disjointGenes / max(len(self.genes), len(genome.genes))

    def evaluateNetwork(self, inpuTT):
        for i in range(Pool.INPUTS):
            self.network[i].value = inpuTT[i]

        for key, val in self.network.items():
            if (key < Pool.INPUTS + Pool.OUTPUTS):
                continue

            neuron = val
            sumWV = 0.0
            for incoming in neuron.inputs:
                other = self.network[incoming.inpuT]
                sumWV += incoming.weight * other.value

            if len(neuron.inputs) != 0:
                neuron.value = Neuron.sigmoid(sumWV)

        for key, val in self.network.items():
            if key < Pool.INPUTS or key >= Pool.INPUTS + Pool.OUTPUTS:
                continue

            neuron = val
            sumWV = 0.0
            for incoming in neuron.inputs:
                other = self.network[incoming.inpuT]
                sumWV += incoming.weight * other.value

            if len(neuron.inputs) != 0:
                neuron.value = Neuron.sigmoid(sumWV)

        output = []
        for i in range(Pool.OUTPUTS):
            output.append(self.network[Pool.INPUTS + i].value)

        return output

    def generateNetwork(self):
        self.network = {}

        for i in range(Pool.INPUTS):
            self.network[i] = Neuron()

        for i in range(Pool.OUTPUTS):
            self.network[Pool.INPUTS + i] = Neuron()

        def getKey(custom):
            return custom.output

        self.genes = sorted(self.genes, key=getKey)

        for gene in self.genes:
            if gene.enabled:
                if gene.output not in self.network.keys():
                    self.network[gene.output] = Neuron()
                neuron = self.network[gene.output]
                neuron.inputs.append(gene)
                if gene.inpuT not in self.network.keys():
                    self.network[gene.inpuT] = Neuron()

    def mutate(self):
        for i in range(7):
            if random.choice([True, False]):
                self.mutationRates[i] *= 0.95
            else:
                self.mutationRates[i] *= 1.05263

        if random.uniform(0, 1) < self.mutationRates[0]:
            self.mutatePoint()

        prob = self.mutationRates[1]
        while (prob > 0):
            if random.uniform(0, 1) < prob:
                self.mutateLink(False)
            prob -= 1

        prob = self.mutationRates[2]
        while (prob > 0):
            if random.uniform(0, 1) < prob:
                self.mutateLink(True);
            prob -= 1

        prob = self.mutationRates[3]
        while (prob > 0):
            if random.uniform(0, 1) < prob:
                self.mutateNode()
            prob -= 1

        prob = self.mutationRates[4]
        while (prob > 0):
            if random.uniform(0, 1) < prob:
                self.mutateEnableDisable(True)
            prob -= 1

        prob = self.mutationRates[5]
        while (prob > 0):
            if random.uniform(0, 1) < prob:
                self.mutateEnableDisable(False)
            prob -= 1

    def mutateEnableDisable(self, enable):
        candidates = []
        for gene in self.genes:
            if gene.enabled != enable:
                candidates.append(gene)

        if len(candidates) == 0:
            return

        gene = candidates[random.randint(0, len(candidates) - 1)]
        gene.enabled = not gene.enabled

    def mutateLink(self, forceBias):
        neuron1 = self.randomNeuron(False, True)
        neuron2 = self.randomNeuron(True, False)

        newLink = Synapse()
        newLink.inpuT = neuron1
        newLink.output = neuron2

        if forceBias:
            newLink.inpuT = Pool.INPUTS - 1;

        if self.containsLink(newLink):
            return

        Pool.innovation += 1
        newLink.innovation = Pool.innovation
        newLink.weight = random.uniform(0, 1) * 4.0 - 2.0

        self.genes.append(newLink)

    def mutateNode(self):
        if len(self.genes) == 0:
            return

        gene = self.genes[random.randint(0, len(self.genes) - 1)]
        if not gene.enabled:
            return
        gene.enabled = False

        self.maxNeuron += 1

        gene1 = gene.clone()
        gene1.output = self.maxNeuron
        gene1.weight = 1.0

        Pool.innovation += 1
        gene1.innovation = Pool.innovation

        gene1.enabled = True
        self.genes.append(gene1)

        gene2 = gene.clone()
        gene2.inpuT = self.maxNeuron

        Pool.innovation += 1
        gene2.innovation = Pool.innovation

        gene2.enabled = True
        self.genes.append(gene2)

    def mutatePoint(self):
        for gene in self.genes:
            if random.uniform(0, 1) < Pool.PERTURBATION:
                gene.weight += random.uniform(0, 1) * self.mutationRates[6] * 2.0 - self.mutationRates[6]
            else:
                gene.weight = random.uniform(0, 1) * 4.0 - 2.0

    def randomNeuron(self, nonInput, nonOutput):
        neurons = []

        if not nonInput:
            for i in range(Pool.INPUTS):
                neurons.append(i)

        if not nonOutput:
            for i in range(Pool.OUTPUTS):
                neurons.append(Pool.INPUTS + i)

        for gene in self.genes:
            if (not nonInput or gene.inpuT >= Pool.INPUTS) and (
                        not nonOutput or gene.inpuT >= Pool.INPUTS + Pool.OUTPUTS):
                neurons.append(gene.inpuT)

            if (not nonInput or gene.output >= Pool.INPUTS) and (
                        not nonOutput or gene.output >= Pool.INPUTS + Pool.OUTPUTS):
                neurons.append(gene.output)
        return neurons[random.randint(0, len(neurons) - 1)]

    def sameSpecies(self, genome):
        dd = Pool.DELTA_DISJOINT * self.disjoint(genome)
        dw = Pool.DELTA_WEIGHTS * self.weights(genome)
        return dd + dw < Pool.DELTA_THRESHOLD

    def weights(self, genome):
        sumW = 0.0
        coincident = 0.0
        for gene in self.genes:
            for otherGene in genome.genes:
                if gene.innovation == otherGene.innovation:
                    sumW += abs(gene.weight - otherGene.weight)
                    coincident += 1.0
                    # print(coincident)
                    break

        if coincident == 0:
            return sumW
        return sumW / coincident
