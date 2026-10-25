# How to present the PatchTST report

This is a simple speaking guide for four people. It explains the paper first, then the work we did. Each person can take one part. The names on the cover still need to be filled in.

## First understand the paper

A **time series** is a list of values in time order, like hourly temperature or electricity use. **Forecasting** means using old values to guess future values. In this paper the forecast can be far ahead, such as 96 or 720 time steps.

The paper is **“A Time Series is Worth 64 Words: Long-term Forecasting with Transformers”** by Nie and others, published at ICLR 2023. The model is called **PatchTST**. Its main question is: can a Transformer predict long time series better if we give it groups of values instead of one value at a time?

It uses two main ideas:

1. **Patching:** Put 16 nearby values into one patch, moving 8 values for the next patch. Each patch is like one token for the Transformer. This makes the input shorter.
2. **Channel independence:** A dataset may have many variables, or channels. The model looks at each variable separately, but uses the same learned weights for all of them. So one model is shared.

The paper also tries **masked pre-training**. Some patches are hidden and the model tries to fill them. After that, it can be used for forecasting.

The goal of our project was to see if the paper's main results can be reproduced on one small laptop GPU. We compared PatchTST with **DLinear**, a simple forecasting baseline. We also removed patching or channel independence to see what happens. We did not try to reproduce every experiment in the paper.

**Two numbers to know:** MSE is mean squared error and MAE is mean absolute error. Both measure forecast mistakes. **Lower is better.** The forecast horizon is how far ahead we predict. Look-back is how much past data the model gets to see.

## Four-person speaking plan

### Person 1: paper and problem, about 1 minute

Say something like:

> We are working on long-term time series forecasting. If we have past weather or electricity values, we want to predict the next many values. A normal Transformer can be expensive when it reads a very long history. The PatchTST paper tries to solve this with patches and a shared model for each variable. Our question was whether its results still come out close on our laptop GPU.

Point to the **Abstract** and **Problem Statement** in the report. Explain the words time series, horizon, and look-back before moving on.

### Person 2: method and data, about 1 to 1.5 minutes

Say something like:

> Here is the basic flow: past values go into 16-step patches. The patches go through one shared Transformer. Then it predicts future values. Each channel is handled separately, but the model weights are shared. We used the authors' code, with small fixes for current libraries. We tested four ETT datasets, Weather and Electricity.

Point to the architecture line in the report. Mention ETT has 7 variables, Weather 21, and Electricity 321. The main look-back is 336 steps. Some training was shorter than in the paper because of our time limit. Do not say our settings are exactly the paper's settings.

### Person 3: main results, about 1.5 minutes

Show **Table 1**. Start with ETTh1 at the 96-step horizon: our PatchTST MSE is **0.376**, the paper has **0.375**, and our DLinear is **0.438**. The other ETT rows also show close paper scores. Weather is **0.155** for us and **0.152** in the paper. Explain that lower is better.

Then say:

> We tested why the method works. With both patching and channel independence, ETTh1 MSE was 0.376. Without patching it was 0.387, without channel independence 0.416, and without both 0.484. So removing these parts made it worse in our test.

Show **Figure 1**. The line goes down from **0.395** at look-back 96 to around **0.375** at 512 and 720. Say the gain becomes small after 336. Do not claim it keeps dropping strongly at every point.

### Person 4: extra test, limits and closing, about 1 minute

Say something like:

> We made one extra Electricity test. We trained PatchTST using only 32 of its 321 channels, but tested it on all 321. It got 0.145 MSE. Training on all channels gave 0.137. This shows the shared model can forecast channels that were not used in training. But the two runs also used different training windows and time, so we cannot say the whole difference is only from the number of channels.

Mention the short masked pre-training result: linear probe on ETTh1/96 got **0.368 MSE**, close to the paper's **0.371**. End with: **the main ETT results came close to the paper, and our ablation gave the same overall trend.** Full Electricity needed a time cap, and the optional Traffic experiment was not run.

## If someone asks a question

- **Why patches?** Fewer tokens go into attention, so long history is easier to handle. A patch also keeps nearby values together.
- **What does channel independence mean?** One variable goes through the model at a time, but all variables share the same model weights.
- **Why use DLinear?** It is a simple baseline. It tells us whether PatchTST is doing better than a much simpler method.
- **Are your results identical to the paper?** No. They are close for most ETT and Weather settings. We used fewer epochs, fewer training windows for some datasets, and mixed precision for larger data.
- **What did four people do?** Present the work as a group, but only claim specific coding or experiment tasks for a member if that member actually did them.

Keep the talk around **5 to 6 minutes**. Speak slowly at the table and graph. The full numbers, settings and paper comparison are in `results/comparison.md` if a teacher asks for more detail.
