import os
import re
from amr_utils.amr_readers import PENMAN_Wrapper, Matedata_Parser
from amr_utils.alignments import AMR_Alignment, write_to_json, load_from_json
from amr_utils.amr import AMR


class AMR_Reader:
    def __init__(self, style="isi"):
        self.style = style

    def load_from_sents(self, sents, remove_wiki=False, output_alignments=False):
        amrs = []
        alignments = {}
        penman_wrapper = PENMAN_Wrapper(style=self.style)
        metadata_parser = Matedata_Parser()
        amr_idx = 0
        no_tokens = False
        if all(sent.strip().startswith("(") for sent in sents):
            no_tokens = True

        for sent in sents:
            prefix_lines = [
                line
                for i, line in enumerate(sent.split("\n"))
                if line.strip().startswith("#") or (i == 0 and not no_tokens)
            ]
            prefix = "\n".join(prefix_lines)
            amr_string_lines = [
                line
                for i, line in enumerate(sent.split("\n"))
                if not line.strip().startswith("#") and (i > 0 or no_tokens)
            ]
            amr_string = "".join(amr_string_lines).strip()
            amr_string = re.sub(" +", " ", amr_string)
            if not amr_string:
                continue
            if not amr_string.startswith("(") or not amr_string.endswith(")"):
                raise Exception("Could not parse AMR from: ", amr_string)
            metadata, graph_metadata = metadata_parser.readlines(prefix)
            tokens = metadata["tok"] if "tok" in metadata else metadata["snt"].split()
            tokens = self._clean_tokens(tokens)
            if graph_metadata:
                amr, aligns = self._parse_amr_from_metadata(tokens, graph_metadata)
                amr.id = metadata["id"]
                if output_alignments:
                    alignments[amr.id] = aligns
            else:
                amr, other_stuff = penman_wrapper.parse_amr(tokens, amr_string)
                if "id" in metadata:
                    amr.id = metadata["id"]
                else:
                    amr.id = str(amr_idx)
                if output_alignments:
                    alignments[amr.id] = []
                    # if "alignments" in metadata:
                    #    aligns = metadata["alignments"].split()
                    #    if any("|" in a for a in aligns):
                    #        jamr_labels = other_stuff[1]
                    #        alignments[amr.id] = self._parse_jamr_alignments(
                    #            amr, amr_file_name, aligns, jamr_labels, metadata_parser
                    #        )
                    #    else:
                    #        isi_labels, isi_edge_labels = other_stuff[2:4]
                    #        alignments[amr.id] = self._parse_isi_alignments(
                    #            amr, amr_file_name, aligns, isi_labels, isi_edge_labels
                    #        )
                    # else:
                    aligns = other_stuff[4]
                    alignments[amr.id] = aligns
            amr.metadata = {k: v for k, v in metadata.items() if k not in ["tok", "id"]}
            amrs.append(amr)
            amr_idx += 1
        if remove_wiki:
            for amr in amrs:
                wiki_nodes = []
                wiki_edges = []
                for s, r, t in amr.edges.copy():
                    if r == ":wiki":
                        amr.edges.remove((s, r, t))
                        del amr.nodes[t]
                        wiki_nodes.append(t)
                        wiki_edges.append((s, r, t))
                if alignments and amr.id in alignments:
                    for align in alignments[amr.id]:
                        for n in wiki_nodes:
                            if n in align.nodes:
                                align.nodes.remove(n)
                        for e in wiki_edges:
                            if e in align.edges:
                                align.edges.remove(e)
        if output_alignments:
            return amrs, alignments
        return amrs

    def load(self, amr_file_name, remove_wiki=False, output_alignments=False):
        print("[amr]", "Loading AMRs from file:", amr_file_name)
        # amrs = []
        # alignments = {}
        # penman_wrapper = PENMAN_Wrapper(style=self.style)
        # metadata_parser = Matedata_Parser()

        with open(amr_file_name, "r", encoding="utf8") as f:
            sents = f.read().replace("\r", "").split("\n\n")
            return self.load_from_sents(sents, remove_wiki, output_alignments)

    def load_from_dir(self, dir, remove_wiki=False, output_alignments=False):
        all_amrs = []
        all_alignments = {}

        taken_ids = set()
        for filename in os.listdir(dir):
            if filename.endswith(".txt"):
                print(filename)
                file = os.path.join(dir, filename)
                amrs, aligns = self.load(file, output_alignments=True, remove_wiki=remove_wiki)
                for amr in amrs:
                    if amr.id.isdigit():
                        old_id = amr.id
                        amr.id = filename + ":" + old_id
                        aligns[amr.id] = aligns[old_id]
                        del aligns[old_id]
                for amr in amrs:
                    if amr.id in taken_ids:
                        old_id = amr.id
                        amr.id += "#2"
                        if old_id in aligns:
                            aligns[amr.id] = aligns[old_id]
                            del aligns[old_id]
                    taken_ids.add(amr.id)
                all_amrs.extend(amrs)
                all_alignments.update(aligns)
        if output_alignments:
            return all_amrs, all_alignments
        return all_amrs

    @staticmethod
    def write_to_file(output_file, amrs):
        with open(output_file, "w+", encoding="utf8") as f:
            for amr in amrs:
                f.write(amr.amr_string())

    @staticmethod
    def load_alignments_from_json(json_file, amrs=None):
        return load_from_json(json_file, amrs=amrs)

    @staticmethod
    def save_alignments_to_json(json_file, alignments):
        write_to_json(json_file, alignments)

    @staticmethod
    def _parse_jamr_alignments(amr, amr_file, aligns, jamr_labels, metadata_parser):
        aligns = [
            (metadata_parser.get_token_range(a.split("|")[0]), a.split("|")[-1].split("+")) for a in aligns if "|" in a
        ]

        alignments = []
        for toks, components in aligns:
            if not all(n in jamr_labels for n in components) or any(t >= len(amr.tokens) for t in toks):
                raise Exception("Could not parse alignment:", amr_file, amr.id, toks, components)
            nodes = [jamr_labels[n] for n in components]
            new_align = AMR_Alignment(type="jamr", tokens=toks, nodes=nodes)
            alignments.append(new_align)
        return alignments

    @staticmethod
    def _parse_isi_alignments(amr, amr_file, aligns, isi_labels, isi_edge_labels):
        aligns = [(int(a.split("-")[0]), a.split("-")[-1]) for a in aligns if "-" in a]

        alignments = []
        xml_offset = 1 if amr.tokens[0].startswith("<") and amr.tokens[0].endswith(">") else 0
        if any(t + xml_offset >= len(amr.tokens) for t, n in aligns):
            xml_offset = 0

        for tok, component in aligns:
            tok += xml_offset
            nodes = []
            edges = []
            if component.replace(".r", "") in isi_labels:
                # node or attribute
                n = isi_labels[component.replace(".r", "")]
                if n == "ignore":
                    continue
                nodes.append(n)
                if n not in amr.nodes:
                    raise Exception("Could not parse alignment:", amr_file, amr.id, tok, component)
            elif not component.endswith(".r") and component not in isi_labels and component + ".r" in isi_edge_labels:
                # reentrancy
                e = isi_edge_labels[component + ".r"]
                edges.append(e)
                if e not in amr.edges:
                    raise Exception("Could not parse alignment:", amr_file, amr.id, tok, component)
            elif component.endswith(".r"):
                # edge
                e = isi_edge_labels[component]
                if e == "ignore":
                    continue
                edges.append(e)
                if e not in amr.edges:
                    raise Exception("Could not parse alignment:", amr_file, amr.id, tok, component)
            elif component == "0.r":
                nodes.append(amr.root)
            else:
                raise Exception("Could not parse alignment:", amr_file, amr.id, tok, component)
            if tok >= len(amr.tokens):
                raise Exception("Could not parse alignment:", amr_file, amr.id, tok, component)
            new_align = AMR_Alignment(type="isi", tokens=[tok], nodes=nodes, edges=edges)
            alignments.append(new_align)
        return alignments

    @staticmethod
    def _parse_amr_from_metadata(tokens, metadata):
        """
        Metadata format is ...
        # ::id sentence id
        # ::tok tokens...
        # ::node node_id node alignments
        # ::root root_id root
        # ::edge src label trg src_id trg_id alignments
        amr graph
        """
        amr = AMR(tokens=tokens)
        alignments = []

        nodes = metadata["node"]
        edges = metadata["edge"] if "edge" in metadata else []
        root = metadata["root"][0]
        amr.root = root[0]
        for data in nodes:
            n, label = data[:2]
            if len(data) > 2:
                toks = data[2]
                alignments.append(AMR_Alignment(type="jamr", nodes=[n], tokens=toks))
            amr.nodes[n] = label
        for data in edges:
            _, r, _, s, t = data[:5]
            if len(data) > 5:
                toks = data[5]
                alignments.append(AMR_Alignment(type="jamr", edges=[(s, r, t)], tokens=toks))
            if not r.startswith(":"):
                r = ":" + r
            amr.edges.append((s, r, t))
        return amr, alignments

    @staticmethod
    def _clean_tokens(tokens):
        line = " ".join(tokens)
        if "<" in line and ">" in line:
            tokens_reformat = []
            is_xml = False
            for i, tok in enumerate(tokens):
                if is_xml:
                    tokens_reformat[-1] += "_" + tok
                    if ">" in tok:
                        is_xml = False
                else:
                    tokens_reformat.append(tok)
                    if tok.startswith("<") and not ">" in tok:
                        if len(tok) > 1 and (tok[1].isalpha() or tok[1] == "/"):
                            if i + 1 < len(tokens) and "=" in tokens[i + 1]:
                                is_xml = True
            tokens = tokens_reformat
        return tokens
