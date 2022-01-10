// NeuralNet.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ******************* class Neuron ******************* //

class Neuron {
public:
    Neuron(unsigned n_outputs, unsigned index);
    void setOutputValue(const double values);
    double getOutputValue(void) const { return m_outputValue; }
    void feedForward(const Layer &previousLayer);
    void computeOuputLayerGradient(const double targetValue);
    void computeHiddenLayerGradient(const Layer &nextLayer);
    void updateInputWeights(Layer &previousLayer);

private:
    static double eta; // [0.0 ... 1.0]
    static double alpha; // [0.0 ... n]
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double transferFunction(const double x);
    static double transferFunctionDerivative(const double x);
    double sumDOW(const Layer &nextLayer) const;
    double m_outputValue;
    unsigned m_index;
    double m_gradient;
    vector<Connection> m_outputWeights;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer& previousLayer) {
    // The weights to be updated are those in the previous layer
    for (unsigned n = 0; n < previousLayer.size(); ++n) {
        Neuron& neuron = previousLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[n].deltaWeight;
        
        double newFDeltaWeight =
            // eta = learning rate
            /*
            * 0.0 --- slow learner
            * 0.2 --- medium learner
            * 1.0 --- reckless learner
            */
            eta
            * neuron.getOutputValue()
            * m_gradient
            // momentum factor which is a fraction of the previous delta
            /*
            * 0.0 --- no momentum
            * 0.5 --- moderate momentum
            */
            +alpha
            * oldDeltaWeight;

        neuron.m_outputWeights[n].deltaWeight = newFDeltaWeight;
        neuron.m_outputWeights[n].weight += newFDeltaWeight;
    }
}


double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    // Sum of contributions of the errors at the node we feed to the next layer
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::computeOuputLayerGradient(const double targetValue) {
    double delta = targetValue - m_outputValue;
    m_gradient = delta * transferFunctionDerivative(m_outputValue);
}

void Neuron::computeHiddenLayerGradient(const Layer& nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputValue);
}



double Neuron::transferFunction(const double x) {
    // Here we use the tanh - output range [-1.0 ; 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(const double x) {
    // Here we use the tanh derivative :
    // The derivative is 1 - tanh²(x) that we can approximate to 1 - x²
    return 1.0 - (x * x);
}

void Neuron::feedForward(const Layer &previousLayer) {
    double sum = 0.0;

    // Here we also feed the bias neuron
    for (unsigned n = 0; n < previousLayer.size(); ++n) {
        sum += previousLayer[n].m_outputValue *
            previousLayer[n].m_outputWeights[m_index].weight;
    }

    m_outputValue = transferFunction(sum);
}


Neuron::Neuron(unsigned n_outputs, unsigned index) {
    for (unsigned c = 0; c < n_outputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_index = index;
}

// ******************* class Network ******************* //

class NeuralNet {
public:
    NeuralNet(const vector<unsigned> &topology);
    // the inputs here are const because they won't be changed
    void feedForward(const vector<double> &inputValues);
    void backPropagation(const vector<double> &targetValues);
    // the function is const because it won't change the object
    void getResults(vector<double> &resultsValues) const;

private:
    vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

void NeuralNet::getResults(vector<double>& resultsValues) const {
    resultsValues.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultsValues.push_back(m_layers.back()[n].getOutputValue());
    }
}


void NeuralNet::backPropagation(const vector<double> &targetValues) {
    // Compute the overall net error using the Root Mean Squared Error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetValues[n] - outputLayer[n].getOutputValue();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    // Implement a recent average measurement
    m_recentAverageError =
        (m_recentAverageError + m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Compute the output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].computeOuputLayerGradient(targetValues[n]);
    }

    // Compute the hidden layers gradients
    for (unsigned l = m_layers.size() - 2; l > 0; --l) {
        Layer &hiddenLayer = m_layers[l];
        Layer &nextLayer = m_layers[l + 1];

        for (unsigned n = 0; n < hiddenLayer.size() - 1; ++n) {
            hiddenLayer[n].computeHiddenLayerGradient(nextLayer);
        }
    }

    // Update all the connection weights from all neurons
    for (unsigned l = m_layers.size() - 1; l > 0; --l) {
        Layer &layer = m_layers[l];
        Layer &previousLayer = m_layers[l - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(previousLayer);
        }
    }
}

void NeuralNet::feedForward(const vector<double>& inputValues) {
    // -1 for the bias neuron
    assert(inputValues.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputValues.size(); ++i) {
        m_layers[0][i].setOutputValue(inputValues[i]);
    }

    // Forward propagate: the propagation starts at layer 1
    for (unsigned l = 1; l < m_layers.size(); ++l) {
        Layer &previousLayer = m_layers[l - 1];
        // -1 for the bias neuron
        for (unsigned n = 0; n < m_layers[l].size() - 1; ++n) {
            m_layers[l][n].feedForward(previousLayer);
        }
    }
}

NeuralNet::NeuralNet(const vector<unsigned>& topology) {
    unsigned n_layers = topology.size();
    for (unsigned n = 0; n < n_layers; ++n) {
        m_layers.push_back(Layer());
        unsigned n_ouputs = n == n_layers - 1 ? 0 : topology[n + 1];

        // <= because of the extra bias neuron
        for (unsigned m = 0; m <= topology[n]; ++m) {
            m_layers.back().push_back(Neuron(n_ouputs, m));
            cout << "New Neuron" << endl;
        }

        m_layers.back().back().setOutputValue(1.0);
    }
}


int main()
{
    cout << "Hello World!\n";
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    NeuralNet neuralNet(topology);

    vector<double> inputValues;
    neuralNet.feedForward(inputValues);

    vector<double> targetValues;
    neuralNet.backPropagation(targetValues);

    vector<double> resultsValues;
    neuralNet.getResults(resultsValues);
}
