import pandas as pd
import matplotlib.pyplot as plt
from train import load_data
from messages import debug, title, normal, success, verbose, error


def get_thetas():
    filename = "data/results.csv"

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        error("File with thetas not found. Please run train.py first.")
        normal("Price estimation is not possible without thetas. Price Car : 0$")
        title("Exiting...")
        exit()
    # return theta0, theta1
    return df["theta0"][0], df["theta1"][0]


def displayEstimate(x, y, theta0, theta1, mileage, estimated_price):
    plt.title("Relationship between a car mileage and its price", size="medium")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (km)")
    # add margin in top of graph
    plt.subplots_adjust(top=0.85)
    plt.plot(
        [min(x), max(x)],
        [theta0 + theta1 * min(x), theta0 + theta1 * max(x)],
        color="C1",
        label="f(x) = {0}*x + {1}".format(round(theta1, 2), round(theta0, 2)),
    )
    plt.plot(x, y, "o", color="C0")
    plt.stem(
        [mileage],
        [estimated_price],
        bottom=(theta0 + theta1 * max(x)),
        orientation="vertical",
        linefmt="--C2",
        markerfmt="oC2",
    )
    plt.stem(
        [estimated_price],
        [mileage],
        bottom=min(x),
        orientation="horizontal",
        linefmt="--C2",
        markerfmt="oC2",
    )
    # add text between graph and title saying "Price of a car with a mileage of X km is Y$"
    plt.text(
        0.5,
        1.1,
        "Price of a car with a mileage of "
        + str(int(mileage))
        + "km is "
        + str(round(estimated_price, 2))
        + "$",
        size="large",
        color="blue",
        transform=plt.gca().transAxes,
        horizontalalignment="center",
    )
    plt.legend()
    plt.show()


def estimate():
    # load thetas
    theta0, theta1 = get_thetas()
    # ask user for mileage
    mileage = input("Enter mileage (in km): ")
    if not mileage.isdigit():
        error("Mileage must be a positive number")
        exit()
    print(
        "\nBased on the trained model, a car with a mileage of",
        mileage,
        "kilometers would be worth :",
    )
    result = theta0 + (theta1 * float(mileage))
    if result < 0:
        print("0 $")
        result = 0
    else:
        print(f"{result:.2f} $")
    df = load_data()
    displayEstimate(df["mileage"], df["price"], theta0, theta1, float(mileage), result)


if __name__ == "__main__":
    estimate()
