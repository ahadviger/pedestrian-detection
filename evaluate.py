from os import listdir
import re
import sys

input_dir = str(sys.argv[1])

files = sorted(listdir(input_dir))

correct, missed, extra = 0, 0, 0
for i, path in enumerate(files):
    if "annotations.txt" in path:
        true = []
        detected = []

        with open(input_dir + "/" + path) as f:
            for line in f:
                m = re.match(r"(Detected|True)\s+(\d+), (\d+) - (\d+), (\d+)", line)
                if m is not None:
                    if "True" in line:
                        true.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
                    elif "Detected" in line:
                        detected.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))

        found = 0
        used = []
        for t in true:
            for i, d in enumerate(detected):
                if i in used:
                    continue

                x = max(0, min(t[2], d[2]) - max(t[0], d[0]))
                y = max(0, min(t[3], d[3]) - max(t[1], d[1]))
                overlap_area = x * y
                true_area = (t[2] - t[0]) * (t[3] - t[1])
                detected_area = (d[2] - d[0]) * (d[3] - d[1])
                overlap_percentage = float(overlap_area) / true_area
                if overlap_percentage > 0.5:
                    found += 1
                    used.append(i)
                    break
        
        correct += found
        missed += len(true) - found
        extra += len(detected) - found

print sys.argv[2]
print "Tocno oznaceni: " + str(correct)
print "Neoznaceni: " + str(missed)
print "Pogresno oznaceni: " + str(extra)