# Training Plan Generator

Basic Chatbot: https://huggingface.co/spaces/Ashilion/Iris

Training Plan Generator: https://training-plan-generator.streamlit.app/ (the website can go to sleep and it can be slow to load the model)

[The presentation slides](peft_presentation.pdf)

## PEFT LoRA with Unsloth

For the fine-tuning process, we used **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** through the **Unsloth** library.  
The main idea behind PEFT + LoRA is to avoid updating all the model weights. Instead, we inject a small number of trainable low-rank matrices into the model. This drastically reduces memory usage and training time while keeping most of the model frozen.

In practice, our LoRA configuration looked something like this:

- Only a small subset of layers was updated (LoRA adapters)
- The rest of the base model stayed frozen  
- This allows fine-tuning large models with a much smaller compute budget

Then we converted the model to GGUF so it’s optimized to run inference on CPU.

(Saving the model to GGUF directly in the notebook didn’t work on Google Colab because of a memory error, so we saved the 16-bit model instead and converted it to GGUF on our own computer)

## Improving Model Performance




### (a) Model-centric approach

We first focused on the model itself.  
At the beginning, we tuned the hyperparameters **by hand** to see if we could get a better training loss. After that, we tried a more automated approach using **Optuna** to search for better hyperparameters.

We also tested several model sizes to understand how different capacities behave:

- **LLaMA 3B** (our first baseline)
- **LLaMA 1B**
- **Gemma 270M**

The idea was to see how smaller models perform, especially in terms of generalization and overfitting.

However, our fine-tuned models ended up performing **worse than the base model**. This could be due to **overfitting** maybe because the **learning rate was too high**.

Our next steps (but we didn’t have time to run them):

- Try a **smaller learning rate**
- Add **LoRA dropout** to increase regularization



---

### (b) Data-centric approach

We also built a small training plan generator on top of the model, but this part was mainly **prompt engineering**, not actual model improvement.

From the data perspective, we noticed that the dataset we fine-tuned on wasn’t necessarily the best fit for running-related tasks. A clear improvement would be to fine-tune the model on a **more appropriate dataset**, something more aligned with the running world (training plans, running advice, performance tracking, etc.).

---


