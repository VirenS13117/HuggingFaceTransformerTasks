"""Download the student model and HumanEval into the HF cache, then smoke-test generation."""
from distill.contamination import load_humaneval
from distill.format import to_messages
from distill.modeling import generate, load_model, pick_device


def main():
    problems = load_humaneval()
    print(f"HumanEval problems: {len(problems)}")
    device = pick_device()
    model, tokenizer = load_model(device)
    print(f"Loaded model on {device}")
    reply = generate(model, tokenizer, to_messages(problems[0]["prompt"]), max_new_tokens=128)
    print("--- sample generation ---\n" + reply)


if __name__ == "__main__":
    main()
