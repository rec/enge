# Two-input, two-output MIDI daughterboard

Revision 0.1, 2026-10-11. Engineering draft, not a fabrication package.

This circuit connects an original Raspberry Pi Zero to two independent MIDI
inputs and two independent MIDI outputs. One SPI dual-UART IC supplies all four
directions. Four female 3.5 mm TRS Type A sockets are intended to occupy one edge
of a future daughterboard. A user-supplied 9 V centre-negative pedal supply powers
the board and Pi. No other processor is required.

## Files and status

- [midi-daughterboard.kicad_sch](midi-daughterboard.kicad_sch): editable, single-sheet
  KiCad 8-format schematic with embedded symbols and explicit physical pin numbers.
- [schematic.svg](schematic.svg): zoomable visual preview of the same circuit,
  generated from the same component and connection definitions, not a KiCad export.
- [bom.csv](bom.csv): 57 physical components, their specifications, and selected
  manufacturer part numbers. Blank part numbers still require selection.
- [connections.csv](connections.csv): pin-level connections, including explicit
  no-connects and two nonphysical power-source flags excluded from the BOM.
- [verification.md](verification.md): checks performed and remaining checks.

Same-name net labels connect throughout the single sheet. Component numbers are
package pin numbers, except the explicitly provisional power connector J1.
The Pi connector uses physical header numbers, not GPIO numbers. Passive symbols
use IEC resistor rectangles, capacitors, and polarity-marked diodes; ICs and
connectors use boxed pin symbols. The BOM and connection table are review
snapshots: regenerate them or update them together when editing the schematic.

The KiCad file has not yet been opened in KiCad or checked by its electrical-rule
checker. All footprints are intentionally unassigned. PCB layout, manufacturing
outputs, software integration, and compliance certification are outside this draft.
Do not order a PCB or assemble this directly from the draft without independent
electrical review and footprint verification.

## Power

J1 represents an insulated 2.1 mm barrel jack: centre negative, outer barrel
positive. Its symbol pins are 1 = centre, 2 = barrel, 3 = unused switched contact.
These are logical assignments until a specific connector is selected; its actual
pad numbering must then replace these assignments. Insulating the jack matters:
the positive outer barrel must not contact grounded enclosure metal.

The positive lead passes through F1, a 0.5 A hold-current, 24 V PTC, and D1, an
SS14 series reverse-polarity diode. D2 is a unidirectional SMBJ12A TVS on the
protected input rail. Its cathode faces the positive rail. The protection is not a
license to use an 18 V supply, nor a substitute for a current-limited, safety-rated
external supply. PTC trip behaviour depends on temperature and available fault
current; it does not impose an exact 500 mA current ceiling. The selected PTC
rating comes from the [Littelfuse 1206L datasheet](https://www.littelfuse.com/assetdocs/littelfuse-ptc-1206l-datasheet?assetguid=2b6a1515-d4ee-4c83-8bd4-152b4901b8f5).

U1 is a synchronous TPS54302 buck converter. C3 connects BOOT directly to SW,
not to ground. L1 connects SW to 5V. The enable divider avoids connecting EN
directly to 9 V. The power stage follows the manufacturer's 5 V reference design,
with its feedback divider adjusted to 100 kohm / 13.5 kohm, both 0.1%.
These connections and starting component values are based on the
[TI TPS54302 datasheet](https://www.ti.com/lit/ds/symlink/tps54302.pdf).

Calculated DC output:

```text
Nominal: 0.596 V * (1 + 100 / 13.5) = 5.011 V
Minimum: 0.581 V * (1 + 99.9 / 13.5135) = 4.876 V
Maximum: 0.611 V * (1 + 100.1 / 13.4865) = 5.146 V
```

These bounds include reference limits and resistor tolerances only. They do not
include wiring losses, ripple, load transients, or temperature drift of the
resistors. Validate those separately. The inductor needs sufficient saturation
margin for the converter's fault-current limits, not merely the normal Pi load.
Select its actual MPN before layout. Check capacitor DC-bias curves, rather than
treating nominal ceramic capacitance as effective capacitance.

The 5V rail feeds Pi header pins 2 and 4. The Pi supplies the 3V3 rail from pins
1 and 17. Ground connects to every ground pin on the header. The daughterboard
does not make its own 3.3 V supply.

**Use one power source only. Do not connect Pi USB power while pedal power is
connected.** There is no power OR-ing or backfeed protection between those sources.
Powering through the header also bypasses any protection associated with the Pi's
power connector. Test the regulator with a dummy load before attaching the Pi.

A 300 mA pedal supply remains a provisional target, not a verified specification.
Measure boot peaks, simultaneous MIDI activity, and sustained software load before
putting a current requirement on the product. The regulator's 3 A rating is not a
claim that this board, its input diode, or the external supply can deliver 3 A.

## UART, clock and reset

U2 is SC16IS752IPW in TSSOP-28. SPI mode 0 runs at no more than 4 MHz. Its I2C/SPI
selection and SDA pins are grounded as required for SPI operation. CTSA and CTSB
are grounded; hardware and software flow control must be disabled. Eight 100 kohm
pulldowns bias unused GPIO pins without directly shorting a possible output to
ground. Leave modem/GPIO functions unused.

Y1 is a 3.3 V, 16 MHz CMOS oscillator. Its output drives XTAL1; XTAL2 is explicitly
unconnected. Enable is tied high. Both UARTs use prescaler 1 and divisor 32:

```text
16,000,000 / (16 * 32) = 31,250 baud
```

There is no divisor rounding error; oscillator tolerance still applies. Do not
substitute a common 14.7456 MHz UART clock without recalculating the baud rate.
Set each channel to eight data bits, no parity, one stop bit, and enable its FIFO.
Divisor registers need explicit initialization even following reset. Clock and
SPI requirements come from the [NXP SC16IS752 datasheet](https://www.nxp.com/docs/en/data-sheet/SC16IS752_SC16IS762.pdf);
oscillator pins and bypassing come from the [Abracon ASE datasheet](https://abracon.com/Oscillators/ASEseries.pdf).

U3 is TLV803EA29DBZR, the 2.93 V threshold, nominal 200 ms, active-low open-drain
reset supervisor. R6 supplies its reset pull-up. R5 is the UART IRQ pull-up.
The supervisor's exact pinout matters because the family has differently wired
variants; use the specified part. See the [TI TLV803E datasheet](https://www.ti.com/lit/ds/symlink/tlv803e.pdf).

| Pi physical pin | Signal | UART pin |
| --- | --- | --- |
| 19 | GPIO10, SPI0 MOSI | U2.11 |
| 21 | GPIO9, SPI0 MISO | U2.12 |
| 23 | GPIO11, SPI0 SCLK | U2.13 |
| 24 | GPIO8, SPI0 CE0 | U2.10 |
| 22 | GPIO25, active-low IRQ | U2.15 |

The Pi's built-in UART is not used. A future host configuration must describe the
16 MHz clock, SPI limit, and active-low interrupt, and service both UART FIFOs
without overruns. That configuration has not been implemented or tested here.

## MIDI sockets and input isolation

All four sockets are SJ1-3523N, with the
[manufacturer's pin assignments](https://www.mouser.com/datasheet/2/670/sj1_352xn-1779157.pdf).
Their [TRS Type A mapping](https://midi.org/specification-for-trs-adapters-adopted-and-released)
is:

| Socket pin | TRS contact | Equivalent DIN contact | Connection |
| --- | --- | --- | --- |
| 1 | Sleeve | 2 | Inputs: no connect; outputs: GND |
| 2 | Tip | 5 | Current-loop return |
| 3 | Ring | 4 | Current-loop positive |

J3 and J4 are the inputs. Each ring feeds a 220 ohm resistor and an optocoupler
LED anode. The LED cathode returns to the tip. D3 and D4 protect the LEDs against
reverse voltage: diode anode to tip, cathode to LED anode. There is no direct
electrical connection from either input loop to board ground or the other input.

U4 and U5 are onsemi H11L1M inverting Schmitt-trigger optocouplers. Their detector
supplies use 5 V, while their open-collector outputs have 270 ohm pull-ups to
3.3 V. The UART therefore sees 3.3 V logic, not 5 V. LED current turns RX low;
no LED current leaves RX high, the correct idle state. H11L1M's 1.6 mA turn-on
threshold gives margin for nominal MIDI current, and its LED current rating
accommodates both 5 V and low-voltage transmitters. Do not substitute H11L2M or
H11L3M: their thresholds differ. Validate timing and margins over the intended
temperature range. Pinout, threshold and load conditions are from the
[onsemi H11L1M datasheet](https://www.onsemi.com/pdf/datasheet/h11l3m-d.pdf).

Keep the input sleeves and socket metalwork isolated from circuit and chassis
ground. Insulating mounting hardware may be required. Copper and enclosure
layout must preserve the optocoupler isolation boundary. Optocoupler isolation
ratings do not qualify the finished board for mains connection.

## MIDI outputs

J5 and J6 are the outputs. U6, SN74LVC2G07DBVR, runs from 3.3 V and supplies two
noninverting, 5 V-tolerant open-drain outputs. Each ring receives 5 V through
220 ohms. Each tip reaches its driver through another 220 ohms. Each sleeve is
grounded at the transmitting end. Driver outputs have 10 kohm pull-ups to 5 V;
UART TX inputs have 10 kohm pull-ups to 3.3 V for startup bias.

TX high releases the driver and stops loop current. TX low sinks loop current.
The output-resistor values follow the 5 V circuit in the
[MIDI electrical specification](https://midi.org/wp-content/uploads/wpforo/default_attachments/1709416667-ca33-MIDI-10-Electrical-Specification-Update.pdf).
The driver's noninverting behaviour, output tolerance and partial-power-down
protection are specified in the [TI SN74LVC2G07 datasheet](https://www.ti.com/lit/ds/symlink/sn74lvc2g07.pdf).

Use 0.25 W output resistors. At the calculated maximum DC rail, a short from the
ring to ground dissipates approximately 0.122 W in a minimum-tolerance 220 ohm
resistor. This calculation does not qualify arbitrary cabling faults. Test shorts,
TRS insertion transients and ESD before deciding whether further protection is
needed for a commercial product.

## Layout and prototype handoff

The future board should stack above the Pi and put the four TRS sockets on one
edge, labelled IN 1, IN 2, OUT 1, OUT 2. This schematic does not determine socket
spacing, enclosure openings, header height or mechanical clearance.

Keep the buck switching loop compact, its feedback trace away from SW, and its
return separated from sensitive signal routing until the ground plane. Follow the
regulator's layout guidance. Place each bypass capacitor at its associated IC.
Keep the clock trace short and keep the isolated input copper away from grounded
copper, enclosure contacts and mounting hardware.

Remaining engineering work is recorded in verification.md. In particular, specify
the actual power jack, inductor, capacitors and footprints before PCB work. Check
every diode's physical cathode marking against its chosen footprint; schematic
diodes use pin 1 = cathode and pin 2 = anode.

## Additional work beyond the prompt

None. This directory contains the requested circuit draft, its component list,
and supporting review material. No application, firmware, library extraction,
PCB layout or enclosure design has been changed.
