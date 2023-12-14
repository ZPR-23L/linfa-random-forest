use linfa::{error::{Error, Result}, Float, Label, ParamGuard};
use crate::{DecisionTreeValidParams, DecisionTree, RandomForestClassifier};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxFeatures {
    Sqrt,
    Log2,
    Float(f32),
    None
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomForestValidParams<F, L> {
    trees_params: DecisionTreeValidParams<F, L>,
    num_trees: usize, // number of estimators
    bootstrap: bool, // is bootstrapping enabled
    oob_score: bool, // is oob score enabled
    max_samples: Option<f32>, // number of samples to bootstrap
    max_features: MaxFeatures // number of features to bootstrap
}

impl<F: Float, L: Label> RandomForestValidParams<F, L> {
    pub fn trees_params(&self) -> DecisionTreeValidParams<F, L> {
        self.trees_params.clone()
    }

    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    pub fn bootstrap(&self) -> bool {
        self.bootstrap
    }

    pub fn oob_score(&self) -> bool {
        self.oob_score
    }

    pub fn max_samples(&self) -> Option<f32> {
        self.max_samples
    }

    pub fn max_features(&self) -> MaxFeatures {
        self.max_features
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomForestParams<F, L> (RandomForestValidParams<F, L>);

impl<F: Float, L: Label> RandomForestParams<F, L> {
    pub fn new() -> Self {
        Self(RandomForestValidParams {
            trees_params: DecisionTree::params().check().unwrap(),
            num_trees: 100,
            bootstrap: true,
            oob_score: false,
            max_samples: None,
            max_features: MaxFeatures::Sqrt
        })
    }

    pub fn trees_params(mut self, trees_params: DecisionTreeValidParams<F, L>) -> Self {
        self.0.trees_params = trees_params;
        self
    }

    pub fn num_trees(mut self, num_trees: usize) -> Self {
        self.0.num_trees = num_trees;
        self
    }

    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.0.bootstrap = bootstrap;
        self
    }

    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.0.oob_score = oob_score;
        self
    }

    pub fn max_samples(mut self, max_samples: Option<f32>) -> Self {
        self.0.max_samples = max_samples;
        self
    }

    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.0.max_features = max_features;
        self
    }
}

impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L: Label> RandomForestClassifier<F, L> {
    pub fn params() -> RandomForestParams<F, L> {
        RandomForestParams::new()
    }
}

impl<F: Float, L> ParamGuard for RandomForestParams<F, L> {
    type Checked = RandomForestValidParams<F, L>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let MaxFeatures::Float(value) = self.0.max_features {
            if value > 0.0 && value < 1.0 {
                return Ok(&self.0);
            } else {
                return Err(Error::Parameters(format!(
                    "Max features should be in range (0, 1), but was {}",
                    value
                )));
            }
        }
        if let Some(value) = self.0.max_samples {
            if value > 0.0 && value < 1.0 {
                return Ok(&self.0);
            } else {
                return Err(Error::Parameters(format!(
                    "Max samples should be in range (0, 1), but was {}",
                    value
                )));
            }
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
