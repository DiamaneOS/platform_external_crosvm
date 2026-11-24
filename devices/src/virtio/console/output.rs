// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Virtio console device output handling.

use std::io;
use std::io::Read;

use base::error;

use crate::virtio::Queue;
use crate::virtio::Reader;

/// Writes the available data from the reader into the given output queue.
///
/// # Arguments
///
/// * `reader` - The Reader with the data we want to write.
/// * `output` - The output sink we are going to write the data to.
fn process_transmit_request(reader: &mut Reader, output: &mut dyn io::Write) -> io::Result<()> {
    // DiamaneOS: copied in fixed chunks. The guest sets the request's length (up to 4 GiB, its
    // descriptors may name the same memory again), so one buffer of that length is not made.
    let mut data = [0u8; TRANSMIT_CHUNK];
    loop {
        let n = reader.read(&mut data)?;
        if n == 0 {
            break;
        }
        output.write_all(&data[..n])?;
    }
    output.flush()?;
    Ok(())
}

/// DiamaneOS: the most bytes of one request held at a time.
const TRANSMIT_CHUNK: usize = 4096;

/// Processes the data taken from the given transmit queue into the output sink.
///
/// # Arguments
///
/// * `interrupt` - Interrupt used to signal (if required) that the queue has been used
/// * `transmit_queue` - The transmit virtio Queue
/// * `output` - The output sink we are going to write the data into
pub fn process_transmit_queue(transmit_queue: &mut Queue, output: &mut dyn io::Write) {
    let mut needs_interrupt = false;
    while let Some(mut avail_desc) = transmit_queue.pop() {
        if let Err(e) = process_transmit_request(&mut avail_desc.reader, output) {
            error!("console: process_transmit_request failed: {}", e);
        }

        transmit_queue.add_used(avail_desc);
        needs_interrupt = true;
    }

    if needs_interrupt {
        transmit_queue.trigger_interrupt();
    }
}

// DiamaneOS: a long request goes out in bounded pieces.
#[cfg(test)]
mod tests {
    use vm_memory::GuestAddress;
    use vm_memory::GuestMemory;

    use super::*;
    use crate::virtio::create_descriptor_chain;
    use crate::virtio::DescriptorType;

    /// Records what it is given and the largest single write.
    #[derive(Default)]
    struct Recorder {
        data: Vec<u8>,
        largest: usize,
    }

    impl io::Write for Recorder {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.largest = self.largest.max(buf.len());
            self.data.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn transmit_in_bounded_chunks() {
        let memory = GuestMemory::new(&[(GuestAddress(0), 0x20000)]).unwrap();
        let sent: Vec<u8> = (0..0x10000u32).map(|i| (i % 251) as u8).collect();
        memory
            .write_all_at_addr(&sent, GuestAddress(0x1000))
            .unwrap();
        let mut chain = create_descriptor_chain(
            &memory,
            GuestAddress(0),
            GuestAddress(0x1000),
            vec![(DescriptorType::Readable, 0x4000); 4],
            0,
        )
        .expect("create_descriptor_chain failed");
        let mut out = Recorder::default();
        process_transmit_request(&mut chain.reader, &mut out).unwrap();
        assert_eq!(out.data, sent);
        assert!(out.largest <= 4096, "one write of {} bytes", out.largest);
    }
}
