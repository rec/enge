# Revision 0.1 verification

Date: 2026-10-11. This is a draft review record, not electrical sign-off.

## Checks completed

- Reviewed the selected UART, reset supervisor, oscillator, input optocoupler,
  output driver, regulator and TRS connector pin assignments against the linked
  datasheets in README.md. J1's physical pin mapping remains provisional.
- Checked the 16 MHz / 16 / 32 baud calculation: exactly 31,250 baud nominal.
- Calculated the feedback-divider DC range, including reference and resistor
  limits: 4.876 to 5.146 V, nominal 5.011 V. Dynamic behaviour remains unverified.
- Parsed the KiCad S-expression structure and checked unique UUIDs: 653.
- Checked 57 physical component instances and two nonphysical power-source flags.
- Checked all 173 connected pin stubs reach their named labels. There are
  41 named nets and 31 explicit no-connect pins. Every connected net has at least
  two endpoints.
- Checked embedded-symbol pin numbers, names, types and coordinates against the
  draft definitions, and checked instance pin lists. Connection points are on a
  1.27 mm (50 mil) grid.
- Checked both input loop endpoint sets contain only their jack, resistor,
  optocoupler LED and reverse-protection diode. Neither loop includes a ground
  endpoint. Checked LED and reverse-diode polarity.
- Checked the BOM, connection table and SVG against their shared draft definitions.
- Rendered the SVG locally and inspected its layout. This was not a KiCad render.

The structural checks used a temporary local draft-generation/checking script;
it is not part of the product or committed as an additional tool. They check
structure and consistency, not whether the circuit will work electrically.

## Checks not completed

KiCad and kicad-cli are not installed in this environment. The schematic has not
been opened by KiCad, and no KiCad electrical-rule check has been performed.
Successful parsing by the structural checker does not prove KiCad compatibility.
No circuit simulation, PCB design, hardware assembly, Linux driver test, or
physical MIDI interoperability test was performed. No Python files or Python-used
data were changed, so the Python test suite was not run for this hardware-only
change.

## Before PCB design

1. Independently review the circuit and open it in KiCad. Run ERC, inspect every
   diagnostic, and correct any import/format or symbol issues. Do not waive
   diagnostics simply because this draft's structural checks passed.
2. Select J1's actual insulated power jack and change its provisional logical
   pin numbers to the manufacturer's physical pad numbers.
3. Select L1 and actual passives, then assign and verify all footprints, including
   diode polarity, H11L1M DIP pin order, and the SJ1-3523N jack pad arrangement.
4. Recheck input LED current and output sink/load margin across supply, resistor,
   temperature and external MIDI-transmitter tolerances. Verify H11L1M output
   levels and timing for a 5 V detector and 3.3 V / 270 ohm output pull-up.
5. Check capacitor effective capacitance and regulator stability with actual parts.
6. Review PTC derating, diode heating, TVS coordination and external supply limits.
   An input-current rating for a finished unit has not been established.
7. Review the enclosure and socket mounting for accidental input-ground bonding.

## Prototype acceptance checks

1. Use a current-limited bench supply and dummy load first. Check polarity,
   regulator startup, rail voltage, ripple, load steps, shutdown and thermal rise.
2. Verify wrong-polarity behaviour and supply recovery without the Pi attached.
   Test protective fault behaviour using suitable controlled equipment.
3. Attach the Pi using pedal power only. Measure boot and sustained-load current,
   5 V droop, 3.3 V sequencing, reset timing, clock frequency and IRQ operation.
4. Configure both UARTs for 31,250 baud, 8N1 with flow control disabled. Check
   startup produces no spurious MIDI messages and idle output polarity is correct.
5. Exercise all four directions simultaneously with maximum-rate byte streams,
   long SysEx passages and interspersed realtime messages. Check byte integrity,
   FIFO overruns, framing errors, latency and timestamp behaviour. The schematic
   alone does not provide hard-realtime guarantees.
6. Test inputs against representative 5 V and 3.3 V MIDI transmitters, including
   the WX7 if available, using Type A adapters and the longest intended cables.
7. Test outputs against legacy and modern receivers. Confirm loop current, logic
   polarity and receiver margins under worst-case load and supply conditions.
8. Check insertion/removal transients, shorts, ESD, conducted noise, radiated EMI
   and other applicable product-compliance requirements. Add protection only
   when the review or qualification tests show it is required.

These are future acceptance checks, not claims that a prototype passed them.
