use ndarray::{Array1, ArrayBase, Data, Ix2};
use linfa::{Float, Label};
use linfa::traits::PredictInplace;
use crate::DecisionTree;

pub struct RandomForestClassifier<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
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