using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using System.IO;

namespace HiddenMarkovModel
{
    class Program
    {
        static void Main(string[] args)
        {
            TestHiddenMarkovModel();
        }

        static void TestHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            int T = 100;
            int K = 2;

            // set hyperparameters
            Dirichlet ProbInitPriorObs = Dirichlet.Uniform(K);
            Dirichlet[] CPTTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            Gaussian[] EmitMeanPriorObs = Enumerable.Repeat(Gaussian.FromMeanAndVariance(0, 1000), K).ToArray();
            Gamma[] EmitPrecPriorObs = Enumerable.Repeat(Gamma.FromShapeAndScale(1000, 0.001), K).ToArray();

            // sample model parameters
            double[] init = ProbInitPriorObs.Sample().ToArray();
            double[][] trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = CPTTransPriorObs[i].Sample().ToArray();
            }
            double[] emitMeans = new double[K];
            for (int i = 0; i < K; i++)
            {
                emitMeans[i] = EmitMeanPriorObs[i].Sample();
            }
            double[] emitPrecs = new double[K];
            for (int i = 0; i < K; i++)
            {
                emitPrecs[i] = EmitPrecPriorObs[i].Sample();
            }

            // print parameters
            HiddenMarkovModel modelForPrinting = new HiddenMarkovModel(T, K);
            modelForPrinting.SetParameters(init, trans, emitMeans, emitPrecs);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            Discrete initDist = new Discrete(init);
            Discrete[] transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }
            Gaussian[] emitDist = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = Gaussian.FromMeanAndPrecision(emitMeans[i], emitPrecs[i]);
            }

            // calculate order of actual states
            int[] actualStateOrder = argSort(emitDist);
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            int[] actualStates = new int[T];
            double[] emissions = new double[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i-1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }
            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
            HiddenMarkovModel model = new HiddenMarkovModel(T, K);
            model.SetPriors(ProbInitPriorObs, CPTTransPriorObs, EmitMeanPriorObs, EmitPrecPriorObs);
            model.ObserveData(emissions);
            model.initialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.ModelEvidencePosterior);
            Discrete[] mapStatesDistr = model.StatesPosterior;
            int[] mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();

            // calculate order of MAP states
            int[] mapStateOrder = argSort(model.EmitMeanPosterior);
            Console.WriteLine("mapStateOrder");
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = 0;
            for (int i = 0; i < actualStates.Length; i++)
            {
                if (actualStateOrder[actualStates[i]] == mapStateOrder[mapStates[i]])
                {
                    correctStates++;
                }
            }
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();

            Console.WriteLine("------------------\n");
        }

        public static int[] argSort(Gaussian[] EmitMeanPosterior)
        {
            int[] order = new int[EmitMeanPosterior.Length];
            for (int i = 0; i < EmitMeanPosterior.Length; i++)
            {
                for (int j = 0; j < EmitMeanPosterior.Length; j++)
                {
                    if (EmitMeanPosterior[i].GetMean() > EmitMeanPosterior[j].GetMean())
                    {
                        order[i]++;
                    }
                }
            }
            return order;
        }
    }
}
