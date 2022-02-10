import multimodel
import featurescalculator

if __name__ == '__main__':
    main()

def main():
    calculator = featurescalculator.FeaturesCalculator()
    model = multimodel.MultiModel(features = calculator.features)