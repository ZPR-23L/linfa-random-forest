use linfa::Float;
use crate::DecisionTreeValidParams;

pub struct RandomForestValidParams<F, L> {
    pub(crate) trees_parameters: DecisionTreeValidParams<F, L>,
    pub(crate) num_trees: i32, // number of estimators
    pub(crate) bootstrap: bool, // is bootstrapping enabled
    pub(crate) oob_score: bool, // is oob score enabled
    pub(crate) max_samples: Option<i32>, // number of samples to bootstrap
}

impl<F: Float, L> RandomForestValidParams<F, L> {
    pub fn set_trees_parameters(&mut self, trees_parameters: DecisionTreeValidParams<F, L>) {
        self.trees_parameters = trees_parameters;
    }
    pub fn set_num_trees(&mut self, num_trees: i32) {
        self.num_trees = num_trees;
    }
    pub fn set_bootstrap(&mut self, bootstrap: bool) {
        self.bootstrap = bootstrap;
    }
    pub fn set_oob_score(&mut self, oob_score: bool) {
        self.oob_score = oob_score;
    }
    pub fn set_max_samples(&mut self, max_samples: Option<i32>) {
        self.max_samples = max_samples;
    }
}