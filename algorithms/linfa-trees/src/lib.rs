//!
//! # Decision tree learning
//! `linfa-trees` aims to provide pure rust implementations
//! of decison trees learning algorithms.
//!
//! # The big picture
//!
//! `linfa-trees` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
//!
//! Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
//! The goal is to create a model that predicts the value of a target variable by learning simple decision rules
//! inferred from the data features.
//!
//! # Current state
//!
//! `linfa-trees` currently provides an [implementation](DecisionTree) of single-tree fitting for classification
//! and an [implementation](RandomForestClassifier) of a random forest - ensamble of decision trees.
//!

mod decision_trees;
mod random_forest;

pub use decision_trees::*;
pub use random_forest::*;
pub use linfa::error::Result;
