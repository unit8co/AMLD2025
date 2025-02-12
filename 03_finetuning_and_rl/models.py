from pydantic import BaseModel, RootModel


class Source(BaseModel):
    """
    Represents a source paragraph from the insurance policy document.

    Attributes:
        paragraph (str): The actual text content from the policy document
        file_name (str): Name of the source file containing the paragraph
        page (int): Page number where the paragraph appears
        line (int): Line number where the paragraph starts
    """

    paragraph: str
    file_name: str
    page: int
    line: int


class SourceRule(BaseModel):
    """
    Represents a specific rule component with its identifier and description.

    Attributes:
        value (str): Unique identifier for the rule (UUID format)
        description (str): Human-readable description of the rule
    """

    value: str
    description: str


class SourceRuleContainer(BaseModel):
    """
    Container for different types of source rules that define the insurance
    claim context.

    Attributes:
        RISK_EVENT (SourceRule): The event that triggered the insurance claim
        OBJECT_OF_COVERAGE (SourceRule): The insured item or person affected
        LOSS (SourceRule): The type of loss or damage incurred
        INDEMNITY (SourceRule | None): The type of compensation or repair
            offered
        SERVICE (SourceRule | None): Additional services provided as part
            of the claim
    """

    RISK_EVENT: SourceRule
    OBJECT_OF_COVERAGE: SourceRule
    LOSS: SourceRule
    INDEMNITY: SourceRule | None = None
    SERVICE: SourceRule | None = None


class Claim(BaseModel):
    """
    Represents an individual insurance claim with all its details and
    associated rules.

    Attributes:
        description (str): Detailed description of the claim scenario
        explanation (str): Additional explanation or notes about the claim
        coverage (bool): Whether the claim is covered under the policy
        sources (list[Source]): Relevant policy document excerpts supporting
            the claim
        source_rule (SourceRuleContainer): Collection of rules applicable to
            the claim
        limit_unit (Optional[str]): Currency unit for the claim limit (e.g.,
            "GBP")
        limit_amount (Optional[float]): Maximum amount covered for the claim
        limit_targets (list[str]): list of entities to whom the limit applies
    """

    description: str
    explanation: str
    coverage: bool
    sources: list[Source]
    source_rule: SourceRuleContainer
    limit_unit: str | None = None
    limit_amount: float | None = None
    limit_targets: list[str]


class ClaimsDataset(RootModel[list[Claim]]):
    """
    Root model representing a collection of insurance claims.

    The __root__ attribute contains a list of all claims in the dataset.
    This model is typically used to parse the entire JSON file containing
    multiple claims.

    Example:
        ```python
        # Parse from file
        with open("data/claims_dataset_v2_manual.json", "r") as f:
            dataset = ClaimsDataset.model_validate_json(f.read())

        # Access first claim
        first_claim = claims.__root__[0]

        # Check coverage
        is_covered = first_claim.coverage
        ```
    """
    root: list[Claim]


if __name__ == "__main__":
    with open("data/claims_dataset_v2_manual.json", "r") as f:
        dataset = ClaimsDataset.model_validate_json(f.read())
    print(len(dataset.root))
