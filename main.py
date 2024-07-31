from pathlib import Path
from math import sqrt
import os
import shutil
import sys
import cv2

HEAT = [
    "\033[40;37m",
    "\033[41;30m",
    "\033[43;30m",
    "\033[42;30m",
    "\033[46;30m",
    "\033[44;30m",
    "\033[45;30m",
    "\033[47;30m",
]
CLR = "\033[0m"

def print_coords(coords: list):
    coords_print = zip(coords[0::3], coords[1::3], coords[2::3])
    for _1, _2, _3 in coords_print:
        A = f"{_1}"
        if len(A) < 32:
            A += "\t"
        B = f"{_2}"
        if len(B) < 32:
            B += "\t"
        C = f"{_3}"
        print(A + "\t" + B + "\t" + C)


def print_table(table: list[list[int]], width=2):
    maximum = max(map(max, table)) or 1
    coef = 7 / maximum
    for row in table:
        string = ""
        for val in row:
            x = int(val * coef)
            string += (
                f"{HEAT[x]} {val:{width}d}{CLR}"
                if val
                else (HEAT[0] + " " * (width + 1) + CLR)
            )
        print(string)


def convert_box_coords(box):
    X, Y, x, y, w, h = box
    newbox = [X, Y, x, x + w, y, y + h]
    return newbox


def path_to_box(paths: list[Path]) -> list[int, int, int, int, int, int]:
    img_names = [path.stem for path in paths]
    coords_str = [name.split("_") for name in img_names]
    for i, coord in enumerate(coords_str):
        coords_str[i][0] = coord[0].lstrip("X")
        coords_str[i][1] = coord[1].lstrip("Y")
    coords_list = [list(map(int, strings)) for strings in coords_str]
    coords_list = [convert_box_coords(coords) for coords in coords_list]
    return coords_list


def find_matches(input_table: list[list[tuple]]) -> list[list[list]]:
    overlaps_table = []
    for row in input_table:
        overlaps_row = []
        for crops_A, crops_B in row:
            overlapping_image_pair = []
            for box_A in crops_A:
                for box_B in crops_B:
                    if IoU(box_A["box"], box_B["box"]) > 0.2:
                        overlapping_image_pair.append((box_A, box_B))
            overlaps_row.append(overlapping_image_pair)
        overlaps_table.append(overlaps_row)
    return overlaps_table


def table_cell_len(table: list[list[list]]) -> list[list[int]]:
    return [[len(cell) for cell in row] for row in table]


def x_and_y_ratio(box_l, box_r) -> (float, float):
    dx_bl = box_l[1]-box_l[0]
    dy_bl = box_l[3]-box_l[2]
    dx_br = box_r[1]-box_r[0]
    dy_br = box_r[3]-box_r[2]
    x_ratio = min(dx_bl, dx_br) / max(dx_bl, dx_br)
    y_ratio = min(dy_bl, dy_br) / max(dy_bl, dy_br)
    return x_ratio, y_ratio


def main(crops_dir, duplicates_dir, user_review, pix_to_um=1):  # pix_to_um = 0.1725
    files = list(crops_dir.glob("*.png"))
    print(len(files), "crops were found")
    coords_list = path_to_box(files)  # output = list[[imgX, imgY, boxX1, boxX2, boxY1, boxY2]
    real_coords_list = [
        composite_to_real_coords(coord, pix_to_um) for coord in coords_list
    ]

    # Sort crops by picture of origin
    # could be done differently but eh...
    X_coord_set = sorted(list({coord[0] for coord in coords_list}))
    Y_coord_set = sorted(list({coord[1] for coord in coords_list}))
    X_dict = dict(zip(X_coord_set, range(len(X_coord_set))))
    Y_dict = dict(zip(Y_coord_set, range(len(Y_coord_set))))
    print(len(X_coord_set), "different X coordinates were found:\n", X_coord_set)
    print(len(Y_coord_set), "different Y coordinates were found:\n", Y_coord_set)
    print("there are in total", len(X_coord_set) * len(Y_coord_set), "original images")

    # 3D table -> [X, Y, crop number]
    # This is where we store the sorted crops
    crops_table = [
        [[] for i in range(len(X_coord_set))] for j in range(len(Y_coord_set))
    ]
    # tried this but it's a trap :
    # crops_table = [[[]] * len(X_coord_set)] * len(Y_coord_set)
    # every cell of the table refer to the same list because
    # multiplying lists copies reference rather than value

    for file, coords, real_coords in zip(files, coords_list, real_coords_list):
        x_index = X_dict[coords[0]]
        y_index = Y_dict[coords[1]]
        crop = {"file": file, "box": real_coords}
        crops_table[y_index][x_index].append(crop)

    counts, empty_count = table_counts(crops_table)
    print("there are", empty_count, "pictures devoid of crops in the current category:")
    print_table(counts, width=3)

    # find overlaps X-wise
    # make pairs of neighboring images
    horizontal_pairs = [list(zip(row[:-1], row[1:])) for row in crops_table]
    vertical_pairs = transpose([list(zip(row[:-1], row[1:])) for row in transpose(crops_table)])
    asc_diag_pairs = diag_ascending_pairs(crops_table)
    desc_diag_pairs = diag_descending_pairs(crops_table)

    neighbor_tables = [
            (horizontal_pairs, "horizontal"),
            (vertical_pairs, "vertical"),
            (asc_diag_pairs, "diagonal ascending"),
            (desc_diag_pairs, "diagonal descending"),
    ]

    remove_count = 0
    for neighbor_table, table_name in neighbor_tables:
        print(f"{table_name} overlaps :")
        overlaps_table = find_matches(neighbor_table)
        print_table(table_cell_len(overlaps_table), width=3)
        count = move_obvious_duplicates(overlaps_table, duplicates_dir, user_review)
        print(count, "images removed")
        remove_count += count
    print(remove_count, "images removed in total")


def move_obvious_duplicates(table, dup_dir, user_review):
    remove_count = 0
    for row in table:
        for cell in row:
            for box_a, box_b in cell:
                if not (box_a["file"].exists() and box_b["file"].exists()):
                    continue
                x_ratio, y_ratio = x_and_y_ratio(box_a["box"], box_b["box"])
                smallest = min(box_a, box_b, key=lambda x: area(x["box"]))
                if user_review:
                    print_heuristic(x_ratio, y_ratio, box_a, box_b)
                if x_ratio > 0.95 or y_ratio > 0.95:
                    shutil.move(smallest["file"], dup_dir)
                    remove_count += 1
                    continue
                if not user_review:
                    continue
                print(f"move {smallest["file"].name} to duplicates ?"+HEAT[5]+"[y/n]"+CLR)
                cv2.imshow("image A", cv2.imread(str(box_a["file"])))
                cv2.imshow("image B", cv2.imread(str(box_b["file"])))
                key = cv2.waitKey(0)
                while key not in (ord("y"), ord("Y"), ord("n"), ord("N"), 27, 3):  # 27 -> ESC, 3 -> Ctrl+C
                    key = cv2.waitKey(0)
                if key in (ord("y"), ord("Y")):
                    shutil.move(smallest["file"], dup_dir)
                    print(HEAT[6]+f"moved {smallest["file"].name} to duplicates"+CLR)
                    remove_count += 1
                elif key in (27, 3):
                    sys.exit()
    return remove_count


def print_heuristic(x_ratio, y_ratio, box_a, box_b):
    string = f"imgA = {box_a["file"].name}, imgB = {box_b["file"].name}"
    if x_ratio > 0.95 and y_ratio > 0.95:
        print(HEAT[4] + "========SAME========" + CLR, string)
    elif x_ratio > 0.95 or y_ratio > 0.95:
        print(HEAT[3] + " X OR Y SAME (CUT?) " + CLR, string)
    elif x_ratio > 0.8 and y_ratio > 0.8:
        print(HEAT[2] + "    could be     " + HEAT[3] + "[+]" + CLR, string)
    elif x_ratio > 0.8 or y_ratio > 0.8:
        print(HEAT[2] + "    not sure     " + HEAT[1] + "[-]" + CLR, string)
    else:
        print(HEAT[1] + "        CUT ?       "+CLR, string)


def diag_descending_pairs(table: list[list[list]]) -> list[list[list[tuple]]]:
    row_pairs = list(zip(table[:-1], table[1:]))
    output_table = []
    for high_row, low_row in row_pairs:
        output_table.append(list(zip(high_row[:-1],low_row[1:])))
    return output_table


def diag_ascending_pairs(table: list[list[list]]) -> list[list[list[tuple]]]:
    row_pairs = list(zip(table[:-1], table[1:]))
    output_table = []
    for high_row, low_row in row_pairs:
        output_table.append(list(zip(high_row[1:],low_row[:-1])))
    return output_table



def transpose(matrix: list[list]) -> list[list]:
    for row in matrix[1:]:
        assert len(row) == len(matrix[0])
    transposed = [[[] for i in range(len(matrix))] for i in range(len(matrix[0]))]
    for i, _ in enumerate(matrix):
        for j, _ in enumerate(matrix[0]):
            transposed[j][i] = matrix[i][j]
    return transposed


def table_counts(crops_table: list[list[list]]) -> (list[list[int]], int):
    counts = []
    empty_count = 0
    for row in crops_table:
        row_counts = []
        for cell in row:
            count = len(cell)
            if count == 0:
                empty_count += 1
            row_counts.append(count)
        counts.append(row_counts)
    return counts, empty_count


def intersection1D(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def intersection2D(box1, box2):
    inter_x = intersection1D(box1[0], box1[1], box2[0], box2[1])
    if not inter_x:
        return 0
    inter_y = intersection1D(box1[2], box1[3], box2[2], box2[3])
    return inter_x * inter_y


def IoU(box1, box2):
    inter = intersection2D(box1, box2)
    union = area(box1) + area(box2) - inter
    if union == 0:
        print(box1, area(box1), box2, area(box2), inter)
    return inter / union


def area(box):
    return (box[1] - box[0]) * (box[3] - box[2])


def composite_to_real_coords(coords, um_per_pix):
    if isinstance(um_per_pix, (list, tuple)):
        x_um_per_pix, y_um_per_pix = um_per_pix
    else:
        x_um_per_pix = um_per_pix
        y_um_per_pix = um_per_pix
    imgX, imgY, boxX1, boxX2, boxY1, boxY2 = coords
    realX1 = imgX + x_um_per_pix * boxX1
    realX2 = imgX + x_um_per_pix * boxX2
    realY1 = imgY + y_um_per_pix * boxY1
    realY2 = imgY + y_um_per_pix * boxY2
    return [realX1, realX2, realY1, realY2]


if __name__ == "__main__":
    user_reviews_crops = False
    if "-u" in sys.argv:
        sys.argv.remove("-u")
        user_reviews_crops = True
    crops_directory = Path(sys.argv[1])
    if len(sys.argv) > 2:
        duplicates = Path(sys.argv[2])
    else:
        duplicates = crops_directory.parent\
                                    .joinpath(f"{crops_directory.name}_duplicates")
        os.makedirs(duplicates, exist_ok=True)
    main(crops_directory, duplicates, user_reviews_crops)
