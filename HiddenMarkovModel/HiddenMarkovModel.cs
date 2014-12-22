using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using System.IO;

namespace HiddenMarkovModel
{
    /// <summary>
    /// Hidden markov model.
    /// </summary>
    class HiddenMarkovModel
    {
        // Set up emission data
        private double[] EmitData;

        // Set up the ranges
        private Range K;
        private Range T;

        // Set up model variables
        private Variable<int> ZeroState;
        private VariableArray<int> States;
        private VariableArray<double> Emissions;

        // Set up model parameters
        private Variable<Vector> ProbInit;
        private VariableArray<Vector> CPTTrans;
        private VariableArray<double> EmitMean;
        private VariableArray<double> EmitPrec;

        // Set up prior distributions
        private Variable<Dirichlet> ProbInitPrior;
        private VariableArray<Dirichlet> CPTTransPrior;
        private VariableArray<Gaussian> EmitMeanPrior;
        private VariableArray<Gamma> EmitPrecPrior;

        // Set up model evidence (likelihood of model given data)
        private Variable<bool> ModelEvidence;

        // Inference engine
        private InferenceEngine Engine;

        // Set up posteriors
        public Dirichlet ProbInitPosterior;
        public Dirichlet[] CPTTransPosterior;
        public Gaussian[] EmitMeanPosterior;
        public Gamma[] EmitPrecPosterior;
        public Discrete[] StatesPosterior;
        public Bernoulli ModelEvidencePosterior;

        /// <summary>
        /// Initializes a new instance of the <see cref="HiddenMarkovModel.HiddenMarkovModel"/> class.
        /// </summary>
        /// <param name="ChainLength">Chain length.</param>
        /// <param name="NumStates">Number states.</param>
        public HiddenMarkovModel(int ChainLength, int NumStates)
        {
            ModelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(ModelEvidence))
            {
                K = new Range(NumStates).Named("K");
                T = new Range(ChainLength).Named("T");
                // Init
                ProbInitPrior = Variable.New<Dirichlet>().Named("ProbInitPrior");
                ProbInit = Variable<Vector>.Random(ProbInitPrior).Named("ProbInit");
                ProbInit.SetValueRange(K);
                // Trans probability table based on init
                CPTTransPrior = Variable.Array<Dirichlet>(K).Named("CPTTransPrior");
                CPTTrans = Variable.Array<Vector>(K).Named("CPTTrans");
                CPTTrans[K] = Variable<Vector>.Random(CPTTransPrior[K]);
                CPTTrans.SetValueRange(K);
                // Emit mean
                EmitMeanPrior = Variable.Array<Gaussian>(K).Named("EmitMeanPrior");
                EmitMean = Variable.Array<double>(K).Named("EmitMean");
                EmitMean[K] = Variable<double>.Random(EmitMeanPrior[K]);
                EmitMean.SetValueRange(K);
                // Emit prec
                EmitPrecPrior = Variable.Array<Gamma>(K).Named("EmitPrecPrior");
                EmitPrec = Variable.Array<double>(K).Named("EmitPrec");
                EmitPrec[K] = Variable<double>.Random(EmitPrecPrior[K]);
                EmitPrec.SetValueRange(K);

                // Define the primary variables
                ZeroState = Variable.Discrete(ProbInit).Named("z0"); // zero state does not have emission variable
                States = Variable.Array<int>(T);
                Emissions = Variable.Array<double>(T);

                // for block over length of chain
                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;
                    var previousState = States[t - 1];

                    // initial distribution
                    using (Variable.If((t == 0).Named("Initial")))
                    {
                        using (Variable.Switch(ZeroState))
                        {
                            States[T] = Variable.Discrete(CPTTrans[ZeroState]);
                        }
                    }

                    // transition distributions
                    using (Variable.If((t > 0).Named("Transition")))
                    {
                        using (Variable.Switch(previousState))
                        {
                            States[t] = Variable.Discrete(CPTTrans[previousState]);
                        }
                    }                         

                    // emission distribution
                    using (Variable.Switch(States[t]))
                    {
                        Emissions[t] = Variable.GaussianFromMeanAndPrecision(EmitMean[States[t]], EmitPrec[States[t]]);
                    }   
                }
            }

            DefineInferenceEngine();
        }

        /// <summary>
        /// Defines the inference engine.
        /// </summary>
        public void DefineInferenceEngine()
        {
            // Set up inference engine
            Engine = new InferenceEngine(new ExpectationPropagation());
            //Engine = new InferenceEngine(new VariationalMessagePassing());
            //Engine = new InferenceEngine(new GibbsSampling());
            Engine.ShowFactorGraph = true;
            Engine.ShowWarnings = true;
            Engine.ShowProgress = true;
            Engine.Compiler.WriteSourceFiles = true;
            Engine.NumberOfIterations = 15;
            Engine.ShowTimings = true;
            Engine.ShowSchedule = false;

        }

        /// <summary>
        /// Initialises the states randomly.
        /// </summary>
        public void InitialiseStatesRandomly()
        {
            VariableArray<Discrete> zinit = Variable<Discrete>.Array(T);
            zinit.ObservedValue = Util.ArrayInit(T.SizeAsInt, t => Discrete.PointMass(Rand.Int(K.SizeAsInt), K.SizeAsInt));
            States[T].InitialiseTo(zinit[T]);
        }

        /// <summary>
        /// Observes the data.
        /// </summary>
        /// <param name="emitData">Emit data.</param>
        public void ObserveData(double[] emitData)
        {
            // Save data as instance variable
            EmitData = emitData;
            // Observe it
            Emissions.ObservedValue = EmitData;
        }

        /// <summary>
        /// Infers the posteriors.
        /// </summary>
        public void InferPosteriors()
        {
            // for monitoring convergence
            //for (int i = 1; i <= 35; i++ )
            //{
            //    Engine.NumberOfIterations = i;
            //    Console.WriteLine(CPTTransPosterior[0]);
            //}

            //infer posteriors
            CPTTransPosterior = Engine.Infer<Dirichlet[]>(CPTTrans);
            ProbInitPosterior = Engine.Infer<Dirichlet>(ProbInit);
            EmitMeanPosterior = Engine.Infer<Gaussian[]>(EmitMean);
            EmitPrecPosterior = Engine.Infer<Gamma[]>(EmitPrec);
            StatesPosterior = Engine.Infer<Discrete[]>(States);
            ModelEvidencePosterior = Engine.Infer<Bernoulli>(ModelEvidence);
        }

        /// <summary>
        /// Resets the inference.
        /// </summary>
        public void ResetInference()
        {
            // reset observations
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                double emit = Emissions[i].ObservedValue;
                Emissions[i].ClearObservedValue();
                Emissions[i].ObservedValue = emit;
            }
        }

        /// <summary>
        /// Sets the uninformed priors.
        /// </summary>
        public void SetUninformedPriors()
        {
            ProbInitPrior.ObservedValue = Dirichlet.Uniform(K.SizeAsInt);
            CPTTransPrior.ObservedValue = Util.ArrayInit(K.SizeAsInt, k => Dirichlet.Uniform(K.SizeAsInt)).ToArray();
            EmitMeanPrior.ObservedValue = Util.ArrayInit(K.SizeAsInt, k => Gaussian.FromMeanAndVariance(1000, 1000000000)).ToArray();
            EmitPrecPrior.ObservedValue = Util.ArrayInit(K.SizeAsInt, k => Gamma.FromMeanAndVariance(0.1, 100)).ToArray();
        }

        /// <summary>
        /// Sets the priors.
        /// </summary>
        /// <param name="ProbInitPriorParamObs">Prob init prior parameter obs.</param>
        /// <param name="CPTTransPriorObs">CPT trans prior obs.</param>
        /// <param name="EmitMeanPriorObs">Emit mean prior obs.</param>
        /// <param name="EmitPrecPriorObs">Emit prec prior obs.</param>
        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs, Gaussian[] EmitMeanPriorObs, Gamma[] EmitPrecPriorObs)
        {
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
            EmitMeanPrior.ObservedValue = EmitMeanPriorObs;
            EmitPrecPrior.ObservedValue = EmitPrecPriorObs;
        }

        /// <summary>
        /// Sets the parameters.
        /// </summary>
        /// <param name="init">Init.</param>
        /// <param name="trans">Trans.</param>
        /// <param name="emitMeans">Emit means.</param>
        /// <param name="emitPrecs">Emit precs.</param>
        public void SetParameters(double[] init, double[][] trans, double[] emitMeans, double[] emitPrecs)
        {
            // fix parameters
            ProbInit.ObservedValue = Vector.FromArray(init);
            Vector[] v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }
            CPTTrans.ObservedValue = v;
            EmitMean.ObservedValue = emitMeans;
            EmitPrec.ObservedValue = emitPrecs;
        }

        /// <summary>
        /// Sets the parameters to MAP estimates.
        /// </summary>
        public void SetParametersToMAPEstimates()
        {
            Vector[] trans = new Vector[K.SizeAsInt];
            double[] emitMean = new double[K.SizeAsInt];
            double[] emitPrec = new double[K.SizeAsInt];
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                trans[i] = CPTTransPosterior[i].PseudoCount;
                emitMean[i] = EmitMeanPosterior[i].GetMean();
                emitPrec[i] = EmitPrecPosterior[i].GetMean();
            }
            ProbInit.ObservedValue = ProbInitPosterior.PseudoCount;
            CPTTrans.ObservedValue = trans;
            EmitMean.ObservedValue = emitMean;
            EmitPrec.ObservedValue = emitPrec;
        }

        /// <summary>
        /// Prints the prior.
        /// </summary>
        public void PrintPrior()
        {
            Console.WriteLine(ProbInitPrior.ObservedValue);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTransPrior.ObservedValue[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitMeanPrior.ObservedValue[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitPrecPrior.ObservedValue[i]);
            }
        }

        /// <summary>
        /// Prints the parameters.
        /// </summary>
        public void PrintParameters()
        {
            Console.WriteLine(ProbInit.ObservedValue);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTrans.ObservedValue[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitMean.ObservedValue[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitPrec.ObservedValue[i]);
            }
        }

        /// <summary>
        /// Prints the posteriors.
        /// </summary>
        public void PrintPosteriors()
        {
            Console.WriteLine(ProbInitPosterior);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTransPosterior[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitMeanPosterior[i]);
            }
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + EmitPrecPosterior[i]);
            }
        }

        /// <summary>
        /// Hyperparameterses to string.
        /// </summary>
        /// <returns>The to string.</returns>
        public string HyperparametersToString()
        {
            string returnString = "";

            // init
            returnString += ProbInitPrior.ObservedValue.PseudoCount + "\n";
            // trans
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += CPTTransPrior.ObservedValue[i].PseudoCount + "\n";
            }
            // emit mean mean
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += EmitMeanPrior.ObservedValue[i].GetMean() + " ";
            }
            returnString += "\n";
            // emit mean var
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += EmitMeanPrior.ObservedValue[i].GetVariance() + " ";
            }
            returnString += "\n";
            // emit prec shape
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += EmitPrecPrior.ObservedValue[i].Shape + " ";
            }
            returnString += "\n";
            // emit prec shape
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += EmitPrecPrior.ObservedValue[i].GetScale() + " ";
            }
            returnString += "\n";

            return returnString;
        }

        /// <summary>
        /// Prints the states.
        /// </summary>
        public void PrintStates()
        {
            string output = "state, power" + "\n";
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                output += Engine.Infer<Discrete>(States[i]).GetMode() + ", " + EmitData[i] + "\n";
            }
            Console.WriteLine(output);

            output = "state, power" + "\n";
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                output += Engine.Infer<Discrete>(States[i]) + ", " + EmitData[i] + "\n";
            }
            Console.WriteLine(output);
        }

        /// <summary>
        /// Returns a <see cref="System.String"/> that represents the current <see cref="HiddenMarkovModel.HiddenMarkovModel"/>.
        /// </summary>
        /// <returns>A <see cref="System.String"/> that represents the current <see cref="HiddenMarkovModel.HiddenMarkovModel"/>.</returns>
        public override string ToString()
        {
            string output = "";
            Boolean PrintInit = true;
            Boolean PrintTrans = true;
            Boolean PrintEmit = true;
            Boolean PrintStates = true;

            // output init
            if (PrintInit)
            {
                output += "ProbInitPosterior" + ProbInitPosterior + "\n";
            }

            // output trans
            if (PrintTrans)
            {
                for (int i = 0; i < K.SizeAsInt; i++)
                {
                    output += "CPTTransPosterior[" + i + "]" + CPTTransPosterior[i] + "\n";
                }
            }

            // output emit
            if (PrintEmit)
            {
                for (int i = 0; i < K.SizeAsInt; i++)
                {
                    output += "Emit Mean Posterior[" + i + "] " + EmitMeanPosterior[i] + "\n";
                    output += "Emit Prec Posterior[" + i + "] " + EmitPrecPosterior[i] + "\n";
                }
            }

            // output states
            if (PrintStates)
            {
                output += "state, power" + "\n";
                for (int i = 0; i < T.SizeAsInt; i++)
                {
                    Console.WriteLine(StatesPosterior[i]);
                    output += StatesPosterior[i].GetMode() + ", " + Emissions[i].ObservedValue + "\n";
                }
            }

            return output;
        }
    }
}
