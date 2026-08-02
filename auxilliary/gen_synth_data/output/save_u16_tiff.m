function save_u16_tiff(img, filename)

    img = max(img, 0);
    img = min(img, 1);

    u16 = uint16(round(img * 65535));
    imwrite(u16, filename, 'Compression', 'none');

end