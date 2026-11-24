// Copyright 2026 DiamaneOS
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! DiamaneOS: DiskOption's `raw` switch, as virtmgr passes a filesystem VM's disk (android/virtmgr/src/crosvm.rs).

use super::DiskOption;

fn from_block_arg(options: &str) -> Result<DiskOption, String> {
    serde_keyvalue::from_key_values(options).map_err(|e| e.to_string())
}

#[test]
fn params_raw_and_direct() {
    let params = from_block_arg("path=/proc/self/fd/7,ro=false,lock=false,direct=true,raw=true").unwrap();
    assert!(params.raw);
    assert!(params.direct);
    assert!(!params.read_only);
    assert!(!params.lock);
    // Off unless asked for, and only a boolean.
    assert!(!from_block_arg("/some/path.img").unwrap().raw);
    assert!(from_block_arg("/some/path.img,raw=yes").is_err());
}
