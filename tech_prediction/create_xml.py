import xml.etree.ElementTree as ET
import numpy as np
from itertools import tee
import glob
import os

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class InvalidNoteStartError(Exception):
    """Exception raised for invalid note start positions."""
    pass

class NoteAlignmentError(Exception):
    """Exception raised when notes don't align properly."""
    pass

def preprocess_input(notes):
    """
    Preprocess the input notes to adjust start positions based on bar number.
    """
    processed_notes = []
    current_bar = 0
    bar_length = 48
    
    for i, note in enumerate(notes):
        start, duration, pitch, string, fret, technique = note
        
        # If the start position is 0 and it's not the first note, it indicates a new bar
        if start == 0 and processed_notes and notes[i-1][1] != 1:
            current_bar += 1
        
        # Adjust the start position
        adjusted_start = start + (current_bar * bar_length)

        if duration == 1:
            prev_dur_ok = i == 0 or notes[i - 1][1] != 1
            next_dur_ok = i == len(notes) - 1 or notes[i + 1][1] != 1
            if prev_dur_ok and next_dur_ok:
                duration = 0
                notes[i + 1][1] += 1
                notes[i + 1][0] -= 1
        
        processed_notes.append([adjusted_start, duration, pitch, string, fret, technique])
    
    return processed_notes

def preprocess_gt(notes):
    """
    Preprocess the input notes to adjust start positions based on bar number.
    """
    processed_notes = []
    current_bar = 0
    bar_length = 48
    
    for i, note in enumerate(notes):
        start, duration, pitch, string, fret, technique = note
        
        # If the start position is 0 and it's not the first note, it indicates a new bar
        if start == 0 and processed_notes and notes[i-1][1] != 1:
            current_bar += 1
        
        # Adjust the start position
        adjusted_start = start + (current_bar * bar_length)

        if duration == 1:
            prev_dur_ok = i == 0 or notes[i - 1][1] != 1
            next_dur_ok = i == len(notes) - 1 or notes[i + 1][1] != 1
            if prev_dur_ok and next_dur_ok:
                duration = 0
                #notes[i + 1][1] += 1
                notes[i + 1][0] -= 1
        
        processed_notes.append([adjusted_start, duration, pitch, string, fret, technique])
    
    return processed_notes

def find_practical_tied_durations(duration, current_beat):
    practical_ties = []
    subdivisions = [3, 4, 6, 8, 9, 12, 18, 24, 36, 48]
    beat_length = [12, 24, 36, 48]

    if current_beat == 0 and duration in subdivisions:
        return practical_ties

    for i in range(1, duration):
        if i in subdivisions and (duration - i) in subdivisions:
            if (current_beat + i in beat_length) and ((current_beat + duration not in beat_length) or (duration not in beat_length)):
                practical_ties.append((i, duration - i))

    return practical_ties

def process_tied_notes(notes):
    result = []
    current_bar = 0
    bar_length = 48
    previous_end = 0

    for i, note in enumerate(notes):
        start, duration, pitch, string, fret, technique = note

        # Check if the note is at the start of a new bar
        if start >= (current_bar + 1) * bar_length:
            current_bar = start // bar_length

        # Check for invalid start position at the beginning of a bar
        if start % bar_length != 0 and start == current_bar * bar_length:
            raise InvalidNoteStartError(f"Invalid start position {start} for note at the beginning of bar {current_bar + 1}")

        # Check if the current start aligns with the previous note's end
        if i > 0 and start != previous_end:
            raise NoteAlignmentError(f"Note misalignment: previous note ended at {previous_end}, but current note starts at {start}")

        current_beat = start % bar_length

        ties = find_practical_tied_durations(duration, current_beat)

        if ties:
            # Choose the first practical tie
            tie = ties[0]
            first_duration, second_duration = tie

            # Create the first note of the tie
            first_note = [start, first_duration, pitch, string, fret, technique, False]
            result.append(first_note)

            # Create the second note of the tie
            second_start = start + first_duration
            #second_note = [second_start, second_duration, pitch, string, fret, technique, True]
            second_note = [second_start, second_duration, pitch, string, fret, 19, True]
            result.append(second_note)

            previous_end = second_start + second_duration
        else:
            # If no practical tie is found, keep the original note
            result.append(note + [False])
            previous_end = start + duration

    return result

QUARTER_NOTES_PER_MEASURE = 48

dur_map = {# dot, triplet, grace
    0: [0, '32nd', False, False, True],
    1: [1, '32nd', False, True, False],
    2: [2, '16th', False, True, False],
    3: [3, '16th', False, False, False],
    4: [4, 'eighth', False, True, False],
    6: [6, 'eighth', False, False, False],
    8: [8, 'quarter', False, True, False],
    9: [9, 'eighth', True, False, False],
    12: [12, 'quarter', False, False, False],
    18: [18, 'quarter', True, False, False],
    24: [24, 'half', False, False, False],
    36: [36, 'half', True, False, False],
    48: [48, 'whole', False, False, False],
}

tech_map = {
    1: ['rest'],
    2: ['slide'],
    3: ['slide in'],
    4: ['slide out'],
    5: ['full bend'],
    6: ['full bend up and down'],
    7: ['full held bend'],
    8: ['full reverse bend'],
    9: ['half bend'],
    10: ['half bend up and down'],
    11: ['half held bend'],
    12: ['half reverse bend'],
    13: ['trill'],
    14: ['mute'],
    15: ['pull-off'],
    16: ['harmonic'],
    17: ['hammer-on'],
    18: ['tap'],
    19: ['normal'],
    20: ['quarter held bend'],
    21: ['quarter reverse bend'],
    22: ['quarter bend'],
    23: ['mute and slide out'],
    24: ['quarter bend up and down'],
}

def apply_technique(technique_id, pitch, notations, technical, note, next_technique_id, is_last):
    """Adds the appropriate MusicXML technique to the <notations> tag."""
    technique = tech_map.get(technique_id, ['normal'])[0]
    #print(technique, technique_id)

    # Handle specific techniques and map them to MusicXML elements
    if 'bend' in technique:
        bend = ET.SubElement(technical, 'bend')
        bend_value = 'full' if 'full' in technique else 'half' if 'half' in technique else 'quarter'
        ET.SubElement(bend, 'bend-alter').text = '2' if bend_value == 'full' else '1' if bend_value == 'half' else '0.5'

        if 'reverse' in technique:
            ET.SubElement(bend, 'pre-bend')
            ET.SubElement(bend, 'release')
        if 'held bend' in technique:
            ET.SubElement(bend, 'pre-bend')
        if 'up and down' in technique:
            ET.SubElement(bend, 'release')

    elif 'slide in' in technique:
        articulations = ET.SubElement(notations, 'articulations')
        ET.SubElement(articulations, 'scoop')

    elif 'slide out' in technique:
        articulations = ET.SubElement(notations, 'articulations')
        ET.SubElement(articulations, 'falloff')

    elif 'slide' in technique:
        ET.SubElement(technical, 'slide', number='6', type="stop")

    elif 'hammer-on' in technique:
        ET.SubElement(technical, 'hammer-on', number='1', type="stop")

    elif 'pull-off' in technique:
        ET.SubElement(technical, 'pull-off', number='1', type="stop")

    elif 'mute' in technique and pitch != 101:
        #ET.SubElement(technical, 'technical').append(ET.Element('mute'))
        play = ET.SubElement(note, 'play')
        ET.SubElement(play, 'mute').text='palm'

    elif 'trill' in technique:
        pi = ET.ProcessingInstruction('GP', '<root><vibrato type="Slight"/></root>')
        note.append(pi)

    elif 'tap' in technique:
        ET.SubElement(technical, 'tap')

    elif 'harmonic' in technique:
        harmonic = ET.SubElement(technical, 'harmonic')
        ET.SubElement(harmonic, 'artificial')
        ET.SubElement(harmonic, 'base-pitch')
    
    if next_technique_id and not is_last:
        next_technique = tech_map.get(next_technique_id, ['normal'])[0]
        if 'hammer-on' in next_technique:
            ET.SubElement(technical, 'hammer-on', number='1', type="start").text='H'
        elif 'pull-off' in next_technique:
            ET.SubElement(technical, 'pull-off', number='1', type='start').text='P'
        elif 'slide' in next_technique:
            ET.SubElement(technical, 'slide', number='6', type='start')

def create_musicxml(notes):
    root = ET.Element('score-partwise', version='2.0')

    # Part-list and score-part details
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id="P1")
    ET.SubElement(score_part, 'part-name').text = 'Guitar'

    identification = ET.SubElement(root, 'identification')
    encoding = ET.SubElement(identification, 'encoding')
    ET.SubElement(encoding, 'encoding-date').text = '2024-09-25'
    ET.SubElement(encoding, 'software').text = 'Guitar Pro 8.1.3'

    # Defaults (page layout, scaling, etc.)
    defaults = ET.SubElement(root, 'defaults')
    scaling = ET.SubElement(defaults, 'scaling')
    ET.SubElement(scaling, 'millimeters').text = '6.4'
    ET.SubElement(scaling, 'tenths').text = '40'
    
    page_layout = ET.SubElement(defaults, 'page-layout')
    ET.SubElement(page_layout, 'page-height').text = '1850'
    ET.SubElement(page_layout, 'page-width').text = '1310'

    # Part-list
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id="P1")
    ET.SubElement(score_part, 'part-name').text = 'Electric Guitar (Distortion)'
    ET.SubElement(score_part, 'part-abbreviation').text = 'dist.guit.'
    midi_instrument = ET.SubElement(score_part, 'midi-instrument', id="P1")
    ET.SubElement(midi_instrument, 'midi-channel').text = '1'
    ET.SubElement(midi_instrument, 'midi-bank').text = '1'
    ET.SubElement(midi_instrument, 'midi-program').text = '31'
    ET.SubElement(midi_instrument, 'volume').text = '100'
    ET.SubElement(midi_instrument, 'pan').text = '0'
    
    # Part section
    part = ET.SubElement(root, 'part', id="P1")
    
    # Measure structure (example)
    current_measure = ET.SubElement(part, 'measure', number="1")
    ET.SubElement(current_measure, 'print', new_system="yes")
    
    # Attributes (key, time, staves, clef)
    attributes = ET.SubElement(current_measure, 'attributes')
    ET.SubElement(attributes, 'divisions').text = '1'
    
    key = ET.SubElement(attributes, 'key')
    ET.SubElement(key, 'fifths').text = '0'
    ET.SubElement(key, 'mode').text = 'major'
    
    time = ET.SubElement(attributes, 'time')
    ET.SubElement(time, 'beats').text = '4'
    ET.SubElement(time, 'beat-type').text = '4'
    
    # Clefs and staff details
    ET.SubElement(attributes, 'staves').text = '2'
    clef1 = ET.SubElement(attributes, 'clef', number="1")
    ET.SubElement(clef1, 'sign').text = 'G'
    ET.SubElement(clef1, 'line').text = '2'
    
    clef2 = ET.SubElement(attributes, 'clef', number="2")
    ET.SubElement(clef2, 'sign').text = 'TAB'
    ET.SubElement(clef2, 'line').text = '5'

    current_measure_number = 1
    #current_measure = None

    for i, note_data in enumerate(notes):
        start_pos, duration, pitch, string, fret, tech, tied = note_data
        next_tech = notes[i + 1][5] if i + 1 < len(notes) else None
        is_last = i == len(notes) - 1

        # Calculate the measure number based on the start position
        measure_number = (start_pos // QUARTER_NOTES_PER_MEASURE) + 1

        # Create a new measure if the note belongs to a new measure
        if measure_number != current_measure_number:
            current_measure = ET.SubElement(part, 'measure', number=str(measure_number))
            current_measure_number = measure_number

        # If no measure exists yet, create the first one
        # if current_measure is None:
        #     current_measure = ET.SubElement(part, 'measure', number=str(current_measure_number))

        # Convert pitch to step and octave (MIDI to MusicXML pitch representation)
        

        # Create a note element
        note = ET.SubElement(current_measure, 'note')

        if pitch == 100:
            ET.SubElement(note, 'rest')
        else:
            if pitch == 101:
                pitch_element = ET.SubElement(note, 'pitch')
                ET.SubElement(pitch_element, 'step').text = 'X'
                ET.SubElement(pitch_element, 'octave').text = '0'
            else:
                step_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                step = step_map[(pitch % 12)]
                octave = (pitch // 12) - 1  # MIDI pitch to octave (MusicXML)

                # Pitch details
                pitch_element = ET.SubElement(note, 'pitch')
                ET.SubElement(pitch_element, 'step').text = step
                ET.SubElement(pitch_element, 'octave').text = str(octave)

        # Duration (in quarter notes)
        ET.SubElement(note, 'duration').text = str(duration)

        if duration in dur_map:
            note_type, dotted, triplet, grace = dur_map[duration][1], dur_map[duration][2], dur_map[duration][3], dur_map[duration][4]
            ET.SubElement(note, 'type').text = note_type

            # Add dot if the note is dotted
            if dotted:
                ET.SubElement(note, 'dot')

            # Add triplet information if applicable
            if triplet:
                time_modification = ET.SubElement(note, 'time-modification')
                ET.SubElement(time_modification, 'actual-notes').text = '3'
                ET.SubElement(time_modification, 'normal-notes').text = '2'

            # Add grace note if applicable
            if grace:
                ET.SubElement(note, 'grace')

        # Voice, type, and other details
        ET.SubElement(note, 'voice').text = '1'
        #ET.SubElement(note, 'type').text = 'quarter'  # Assumption: treating as quarter note

        # Add notations for string and fret
        if pitch != 100:
            notations = ET.SubElement(note, 'notations')
            technical = ET.SubElement(notations, 'technical')
            ET.SubElement(technical, 'string').text = str(string)
            ET.SubElement(technical, 'fret').text = str(fret)

            apply_technique(tech, pitch, notations, technical, note, next_tech, is_last)

            if pitch == 101:
                #ET.SubElement(technical, 'mute')
                play = ET.SubElement(note, 'play')
                ET.SubElement(play, 'mute').text='straight'

            if tied:
                ET.SubElement(note, 'tie', type='stop')
            if i + 1 < len(notes) and notes[i + 1][-1]:
                tied_note = ET.SubElement(note, 'tie', type='start')

    return ET.ElementTree(root)

def main():    
    
    '''file = #file path'''
    '''path = #output xml path'''
    
    with open(file, 'r') as f:
        lines = [line.strip()[1:-1].split(', ') for line in f.read().split("\n")][:-1]
    f.close()
    pred = [list(map(int, line)) for line in lines]
    preprocessed_notes = preprocess_input(pred)
    try:
        output_notes = process_tied_notes(preprocessed_notes)
        tree = create_musicxml(output_notes)
        ET.indent(tree, space="  ", level=0)  # To make it readable
        tree.write(path, encoding='utf-8', xml_declaration=True)
    except (InvalidNoteStartError, NoteAlignmentError) as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()
