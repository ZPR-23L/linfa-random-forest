use ndarray::{Array1, ArrayBase, Data, Ix2};
use linfa::{
    dataset::{Labels, AsSingleTargets},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};
use linfa::prelude::Fit;
use linfa::traits::PredictInplace;
use crate::{DecisionTree, DecisionTreeParams, RandomForestValidParams};

pub struct RandomForestClassifier<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
    oob_score: Option<f32>
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
for RandomForestClassifier<F, L>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
       // TODO implement
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
        // TODO implement
    }
}

impl<F, L, D, T> RandomForestClassifier<F, L> {
    fn calculate_oob_score() -> Option<f32> {
        // TODO implement
        // TODO correct function signature
        Float::default()
    }
    fn bootstrap(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, num_trees: i32,
    max_samples: i32) -> Vec<&DatasetBase<ArrayBase<D, Ix2>, T>>
        where
            D: Data<Elem = F>,
            T: AsSingleTargets<Elem = L> + Labels<Elem = L> {
        // TODO implement
        Vec::default()
    }
    fn build_trees() -> Vec<DecisionTree<F, T>> {
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

        if !self.bootstrap && self.oob_score {
            Err("OOB score is available only with bootstrap")
        }

        let mut fitted_trees: Vec<DecisionTree<F, T>> = Vec::new();
        if self.bootstrap {
            let samples = self.bootstrap(dataset, self.num_trees, self.max_samples.unwrap());
            for sample in samples {
                let tree = DecisionTreeParams::from(DecisionTreeParams(
                    self.trees_parameters.clone()
                )).fit(sample);
                fitted_trees.push(tree.unwrap())
            }
        }
        else {
            for num_tree in 0..self.num_trees {
                let tree = DecisionTreeParams::from(DecisionTreeParams(
                    self.trees_parameters.clone()
                )).fit(dataset);
                fitted_trees.push(tree.unwrap())
            }
        }

        let oob_score ;
        if self.oob_score {
            oob_score = RandomForestClassifier::calculate_oob_score()
        }
        else { oob_score = None }

        Ok(RandomForestClassifier{
            trees: fitted_trees,
            oob_score
        })
    }

}
