interface FilingsListProps {
  filings: string[];
}

interface ParsedFiling {
  form: string;
  date: string;
  accession: string;
}

function parseFiling(raw: string): ParsedFiling {
  const match = raw.match(/^(\S+)\s+(\S+)\s+\((.+)\)$/);
  if (!match) return { form: raw, date: "", accession: "" };
  const [, form, date, accession] = match;
  return { form, date, accession };
}

export default function FilingsList({ filings }: FilingsListProps) {
  if (filings.length === 0) {
    return (
      <div className="card">
        <h2>Recent Filings</h2>
        <div className="empty-state">No filings retrieved yet.</div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Recent Filings</h2>
      <table>
        <thead>
          <tr>
            <th>Form</th>
            <th>Filed</th>
            <th>Accession Number</th>
          </tr>
        </thead>
        <tbody>
          {filings.map((raw) => {
            const { form, date, accession } = parseFiling(raw);
            return (
              <tr key={raw}>
                <td>{form}</td>
                <td>{date}</td>
                <td>{accession}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
