use std::collections::HashMap;

use ndarray::{Array1, ArrayBase, Data, Ix2};
use linfa::{
    dataset::{Labels, AsSingleTargets},
    error::Error,
    error::Result,
    DatasetBase, Float, Label,
};
use linfa::prelude::Fit;
use linfa::traits::{PredictInplace, Predict};
use crate::{DecisionTree, RandomForestValidParams};

pub struct RandomForestClassifier<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
    oob_score: Option<f32>
}

impl<F: Float, L: Label> RandomForestClassifier<F, L> {
    fn calculate_oob_score() -> Option<f32> {
        // TODO implement
        // TODO correct function signature
        None
    }

    fn bootstrap<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, num_trees: usize,
                             max_samples: f32) -> Vec<&DatasetBase<ArrayBase<D, Ix2>, T>> {
        // TODO implement
        Vec::default()
    }
    fn build_trees<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>() -> Vec<DecisionTree<F, L>> {
        // TODO implement
        Vec::default()
    }
}


impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
for RandomForestValidParams<F, L>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = RandomForestClassifier<F, L>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

        // TODO extend implementation
        // This is a draft - many things may be changed or added

        let mut fitted_trees: Vec<DecisionTree<F, L>> = Vec::new();
        if self.bootstrap() {
            let samples = RandomForestClassifier::bootstrap(dataset, self.num_trees(), self.max_samples().unwrap());
            for sample in samples {
                let tree = DecisionTree::params()
                    .split_quality(self.trees_params().split_quality())
                    .max_depth(self.trees_params().max_depth())
                    .min_weight_split(self.trees_params().min_weight_split())
                    .min_weight_leaf(self.trees_params().min_weight_leaf())
                    .min_impurity_decrease(self.trees_params().min_impurity_decrease())
                    .fit(sample);
                fitted_trees.push(tree.unwrap())
            }
        }
        else {
            for _num_tree in 0..self.num_trees() {
                let tree = DecisionTree::params()
                    .split_quality(self.trees_params().split_quality())
                    .max_depth(self.trees_params().max_depth())
                    .min_weight_split(self.trees_params().min_weight_split())
                    .min_weight_leaf(self.trees_params().min_weight_leaf())
                    .min_impurity_decrease(self.trees_params().min_impurity_decrease())
                    .fit(dataset);
                fitted_trees.push(tree.unwrap())
            }
        }

        let oob_score ;
        if self.oob_score() {
            oob_score = RandomForestClassifier::<F, L>::calculate_oob_score()
        }
        else { oob_score = None }

        Ok(RandomForestClassifier{
            trees: fitted_trees,
            oob_score
        })
    }

}

impl<F: Float, L: Label + Default + Copy, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
for RandomForestClassifier<F, L>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        // key is the row's index and the value is a vector of labels for that row from all trees
        let mut trees_targets: HashMap<usize, Vec<L>> = HashMap::new();

        for tree in &self.trees {
            let targets = tree.predict(x);
            for (idx, target) in targets.iter().enumerate() {
                let row_targets = trees_targets.entry(idx).or_insert(Vec::new());
                row_targets.push(*target);
            }
        }

        // search for most frequent label in each row
        for (idx, target) in y.iter_mut().enumerate() {
            *target = most_common::<L>(trees_targets.get(&idx).unwrap()).clone();
        }

    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

fn most_common<L: std::hash::Hash + Eq>(targets: &[L]) -> &L {
    let mut map = HashMap::new();
    for target in targets {
        let counter = map.entry(target).or_insert(0);
        *counter += 1;
    }
    let (most_common, _) = map.into_iter().max_by_key(|(_, v)| *v).unwrap();
    most_common
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

    #[test]
    fn invalid_bootstrap_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.bootstrap(false).oob_score(true);
        let result = params.check_ref();
        assert!(result.is_err());
    }
}