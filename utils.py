import numpy as np
from datetime import datetime, timedelta


def read_period_file(file):
    """Reads a calendar file and extracts period start and end dates."""
    period_cal = []
    periods = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            newline = line.strip().split("\t")
            if len(newline) < 2:
                continue
            if newline[1] == "Period Starts":
                period_cal.append([datetime.strptime(newline[0], "%d %b, %Y")])
            elif newline[1] == "Period Ends" and period_cal:
                period_cal[-1].append(datetime.strptime(newline[0], "%d %b, %Y"))

    for i in range(1, len(period_cal)):
        start_date = period_cal[i][0]
        prev_start_date = period_cal[i - 1][0]
        end_date = period_cal[i][1]
        cycle_length = (start_date - prev_start_date).days
        menstruation_length = (end_date - start_date).days + 1
        periods.append([start_date, cycle_length, menstruation_length])

    return periods


def make_train_test_sets(periods):
    x = []
    y = []

    for i in range(len(periods) - 3):
        x.append([
            [periods[i][1], periods[i][2]],
            [periods[i + 1][1], periods[i + 1][2]],
            [periods[i + 2][1], periods[i + 2][2]],
        ])
        y.append([periods[i + 3][1], periods[i + 3][2]])

    x *= 5
    y *= 5
    train_size = int(len(y) * 0.8)

    train_x = np.array(x[:train_size])
    train_y = np.array(y[:train_size])
    test_x = np.array(x[train_size:])
    test_y = np.array(y[train_size:])

    last_known_period = (periods * 5)[train_size][0]
    return train_x, train_y, test_x, test_y, last_known_period


def load_synthetic_data(file):
    periods = []
    with open(file, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            if len(values) == 2:
                periods.append([int(v) for v in values])

    x = []
    y = []

    for i in range(len(periods) - 3):
        x.append([periods[i], periods[i + 1], periods[i + 2]])
        y.append(periods[i + 3])

    x *= 5
    y *= 5

    train_size = int(len(y) * 0.8)
    train_x = np.array(x[:train_size])
    train_y = np.array(y[:train_size])
    test_x = np.array(x[train_size:])
    test_y = np.array(y[train_size:])

    return train_x, train_y, test_x, test_y


def evaluate_predictions(test_y, predictions):
    assert len(test_y) == len(predictions)
    right_cycle = sum(1 for i in range(len(test_y)) if test_y[i][0] == predictions[i][0])
    right_menstr = sum(1 for i in range(len(test_y)) if test_y[i][1] == predictions[i][1])
    return right_cycle / len(test_y), right_menstr / len(test_y)


def print_predictions(last_known_period, predictions):
    next_periods = [[
        last_known_period + timedelta(days=predictions[0][0]),
        last_known_period + timedelta(days=predictions[0][0] + predictions[0][1]),
        predictions[0][1]
    ]]

    for period in predictions[1:]:
        last_period = next_periods[-1]
        next_periods.append([
            last_period[0] + timedelta(days=period[0]),
            last_period[0] + timedelta(days=period[0] + period[1]),
            period[1]
        ])

    for num, period in enumerate(next_periods):
        print(f"{num}. From {period[0].strftime('%d.%m.%Y')} to {period[1].strftime('%d.%m.%Y')}, length: {period[2]}")

    return next_periods
