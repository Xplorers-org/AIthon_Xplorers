import utils.transport
import utils.session
from utils.predictor import predict_spiral, predict_wave

def print_result(result):
    print("\n==============================")
    print(f"Test Type: {result['type']}")
    print("==============================")
    print(f"Prediction       : {result['label']}")
    print(f"Logit            : {result['logit']:.4f}")
    print(f"PD Probability   : {result['probability']:.4f}")
    print(f"Severity Score   : {result['score']:.2f}/100")
    print(f"Severity Level   : {result['severity']}")
    print(f"Interpretation   : {result['description']}")
    print("==============================\n")


def main():
    print("🧠 Parkinson Detection System")

    while True:
        print("\nChoose Test Type:")
        print("1. Spiral Test")
        print("2. Wave Test")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            path = input("Enter spiral image path: ")
            result = predict_spiral(path)
            print_result(result)

        elif choice == "2":
            path = input("Enter wave image path: ")
            result = predict_wave(path)
            print_result(result)

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()