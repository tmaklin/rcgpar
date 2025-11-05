// rcgpar: Riemannian conjugate gradient descent for estimating mixture model weights.
//
// Copyright 2025 Tommi Mäklin [tommi@maklin.fi].
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
// USA
//

//! Implementation of the Riemannian conjugate gradient descent algorithm used
//! in Mäklin et al. 2020 in Wellcome Open Research.
//!
//! This implementation is based on the rcg_gpu Torch code written by Joel.
//!

use burn_tensor::backend::Backend;
use burn_tensor::backend::Device;
use burn_tensor::Tensor;

type E = Box<dyn std::error::Error>;

pub fn logsumexp(

) -> Result<(), E> {
    todo!("Implement logsumexp");

    Ok(())
}

pub fn newnorm<B: Backend<FloatElem = f32>>(
    gamma_Z: Tensor::<B, 2>,
    dl_dphi: Tensor::<B, 2>,
) -> Result<(), E> {
    todo!("Implement newnorm");

    Ok(())
}

pub fn mixt_negnatgrad<B: Backend, D: Device>(
    logl: Tensor::<B, 2>,
    gamma_Z: Tensor::<B, 2>,
    n_k: Tensor::<B, 1>,
) -> Result<(), E> {
    todo!("Implement mixt_negnatgrad");

    Ok(())
}

pub fn update_N_k(

) -> Result<(), E> {
    todo!("Implement update_N_k");
    Ok(())
}

pub fn elbo_rcg_mat(

) -> Result<(), E> {
    todo!("Implement ELBO_rcg_mat");
    Ok(())
}

pub fn calc_bound_const(

) -> Result<(), E> {
    todo!("Implement calc_bound_const");
    Ok(())
}

pub fn rcg_optl_mat(

) -> Result<(), E> {
    todo!("Implement rcg_optl_mat");
    Ok(())
}
