use ndarray::{Array1, ArrayBase, Data, Ix2};
use linfa::{Float, Label};
use linfa::traits::PredictInplace;
use crate::DecisionTree;

pub struct RandomForestClassifier<F: Float, L: Label> {
    _trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
for RandomForestClassifier<F, L>
{
    fn predict_inplace(&self, _x: &ArrayBase<D, Ix2>, _y: &mut Array1<L>) {
       // TODO implement
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
        // TODO implement
    }
}


#[cfg(test)]
mod tests {
    use linfa::ParamGuard;
    use crate::{RandomForestClassifier, RandomForestValidParams, RandomForestParams, MaxFeatures};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin> () {}
        has_autotraits::<RandomForestClassifier<f64, bool>>();
        has_autotraits::<RandomForestValidParams<f64, bool>>();
        has_autotraits::<RandomForestParams<f64, bool>>();
    }

    #[test]
    fn default_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let valid_params = params.check().unwrap();
        assert_eq!(valid_params.num_trees(), 100);
        assert_eq!(valid_params.bootstrap(), true);
        assert_eq!(valid_params.oob_score(), false);
        assert_eq!(valid_params.max_samples(), None);
        assert_eq!(valid_params.max_features(), MaxFeatures::Sqrt);
    }

    #[test]
    fn custom_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let valid_params = params.num_trees(50)
            .oob_score(true)
            .max_samples(Some(0.5))
            .max_features(MaxFeatures::None)
            .check()
            .unwrap();
        assert_eq!(valid_params.num_trees(), 50);
        assert_eq!(valid_params.bootstrap(), true);
        assert_eq!(valid_params.oob_score(), true);
        assert_eq!(valid_params.max_samples(), Some(0.5));
        assert_eq!(valid_params.max_features(), MaxFeatures::None);
    }

    #[test]
    fn invalid_max_samples() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_samples(Some(1.5));
        let result = params.check_ref();
        assert!(result.is_err());
    }

    #[test]
    fn invalid_max_features() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_features(MaxFeatures::Float(1.5));
        let result = params.check_ref();
        assert!(result.is_err());
    }
}