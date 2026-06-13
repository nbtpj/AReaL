codex --sandbox danger-full-access "Run the experiment for Vanilla method for RL on dataset Polaris using Qwen3-1.7B. Test on AIME24 dataset just as given in the code. Save the ckpt every 5 steps. Eval every 10 steps. First run for 100 steps, record the test accuracies in vanilla.md, keep the ckpt in directory ckpts/vanilla, Here is a ray server with 2 nodes"

codex --sandbox danger-full-access "Run the experiment for word_shuffle method for RL on dataset Polaris using Qwen3-1.7B. Test on AIME24 dataset just as given in the code. Save the ckpt every 5 steps. Eval every 10 steps. First run for 100 steps, record the test accuracies in word_shuffle.md, keep the ckpt in directory ckpts/word_shuffle, Here is a ray server with 2 nodes"

codex --sandbox danger-full-access "Compare the results (accuracy) in vanilla.md and word_shuffle.md. Analyze the results and write a conclusion in conclusion.md" 

